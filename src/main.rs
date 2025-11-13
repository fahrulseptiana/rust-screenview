use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use clap::Parser;
use pollster::block_on;
use wgpu::util::DeviceExt;
use windows_capture::{
    capture::{CaptureControl, Context as CaptureContext, GraphicsCaptureApiHandler},
    frame::Frame,
    graphics_capture_api::InternalCaptureControl,
    monitor::Monitor,
    settings::{
        ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
        MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
    },
};
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    keyboard::{Key, NamedKey},
    window::{Fullscreen, Window, WindowBuilder},
};

const DOUBLE_CLICK_THRESHOLD: Duration = Duration::from_millis(350);

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "View a Windows display in a low-latency preview window"
)]
struct Args {
    /// Display index (0-based). If omitted you'll be prompted.
    #[arg(short, long)]
    display: Option<usize>,

    /// Target capture FPS (maps to the Graphics Capture minimum update interval).
    #[arg(long, default_value_t = 60)]
    fps: u32,
}

#[derive(Debug, Clone, Copy)]
enum UserEvent {
    FrameReady,
    CaptureClosed,
}

#[derive(Clone)]
struct FramePacket {
    pixels: Arc<[u8]>,
    width: u32,
    height: u32,
}

type SharedFrame = Arc<Mutex<Option<FramePacket>>>;

struct CaptureFlags {
    frame_slot: SharedFrame,
    proxy: EventLoopProxy<UserEvent>,
}

struct FrameDispatcher {
    frame_slot: SharedFrame,
    proxy: EventLoopProxy<UserEvent>,
    scratch: Vec<u8>,
}

impl GraphicsCaptureApiHandler for FrameDispatcher {
    type Flags = CaptureFlags;
    type Error = anyhow::Error;

    fn new(ctx: CaptureContext<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self {
            frame_slot: ctx.flags.frame_slot,
            proxy: ctx.flags.proxy,
            scratch: Vec::new(),
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let buffer = frame.buffer().context("failed to map capture frame")?;
        let contiguous = buffer.as_nopadding_buffer(&mut self.scratch);
        let packet = FramePacket {
            pixels: Arc::from(contiguous),
            width: buffer.width(),
            height: buffer.height(),
        };
        {
            let mut slot = self.frame_slot.lock().unwrap();
            *slot = Some(packet);
        }
        let _ = self.proxy.send_event(UserEvent::FrameReady);
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        let _ = self.proxy.send_event(UserEvent::CaptureClosed);
        Ok(())
    }
}

struct MonitorDescriptor {
    monitor: Monitor,
    name: String,
    width: u32,
    height: u32,
    refresh: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let monitor_handles = Monitor::enumerate().context("unable to enumerate monitors")?;
    if monitor_handles.is_empty() {
        bail!("no monitors detected");
    }

    let mut monitors = Vec::with_capacity(monitor_handles.len());
    for monitor in monitor_handles {
        let name = monitor
            .name()
            .unwrap_or_else(|_| "Unknown monitor".to_string());
        let width = monitor.width().context("failed to read monitor width")?;
        let height = monitor.height().context("failed to read monitor height")?;
        let refresh = monitor.refresh_rate().unwrap_or(0);
        monitors.push(MonitorDescriptor {
            monitor,
            name,
            width,
            height,
            refresh,
        });
    }

    println!("Detected {} display(s):", monitors.len());
    for (idx, info) in monitors.iter().enumerate() {
        println!(
            "  [{}] {}x{} @ {}Hz {}",
            idx, info.width, info.height, info.refresh, info.name
        );
    }

    let selection = resolve_display_index(args.display, monitors.len())?;
    let target = monitors
        .get(selection)
        .ok_or_else(|| anyhow!("display {selection} disappeared"))?;

    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event()
        .build()
        .context("failed to create event loop")?;
    let proxy = event_loop.create_proxy();
    let shared_frame: SharedFrame = Arc::new(Mutex::new(None));

    let capture_flags = CaptureFlags {
        frame_slot: shared_frame.clone(),
        proxy,
    };

    let settings = Settings::new(
        target.monitor,
        CursorCaptureSettings::WithCursor,
        DrawBorderSettings::WithoutBorder,
        SecondaryWindowSettings::Default,
        minimum_update_interval(args.fps),
        DirtyRegionSettings::Default,
        ColorFormat::Rgba8,
        capture_flags,
    );

    let capture_control =
        FrameDispatcher::start_free_threaded(settings).context("failed to start capture")?;
    let capture_control = Arc::new(Mutex::new(Some(capture_control)));

    let window_arc = Arc::new(
        WindowBuilder::new()
            .with_title(format!(
                "ScreenView - {} ({}x{} @ {}Hz)",
                target.name, target.width, target.height, target.refresh
            ))
            .with_inner_size(PhysicalSize::new(
                target.width.max(1) as f64,
                target.height.max(1) as f64,
            ))
            .with_maximized(true)
            .build(&event_loop)
            .context("failed to build window")?,
    );

    window_arc.set_maximized(true);

    let mut gpu = block_on(GpuState::new(window_arc.as_ref()))?;
    let window = window_arc.clone();

    let mut is_fullscreen = window.fullscreen().is_some();
    let mut saved_window_size: Option<PhysicalSize<u32>> = None;
    let mut saved_window_position: Option<PhysicalPosition<i32>> = None;
    let mut last_click: Option<Instant> = None;

    let mut latest_capture_size: Option<(u32, u32)> = None;

    event_loop
        .run(move |event, elwt| {
            let window = window.as_ref();
            match event {
                Event::WindowEvent { window_id, event } if window_id == window.id() => {
                    match event {
                        WindowEvent::CloseRequested => {
                            stop_capture(&capture_control);
                            elwt.exit();
                        }
                        WindowEvent::MouseInput {
                            state: ElementState::Pressed,
                            button: MouseButton::Left,
                            ..
                        } => {
                            let now = Instant::now();
                            if last_click
                                .map(|prev| now.duration_since(prev) <= DOUBLE_CLICK_THRESHOLD)
                                .unwrap_or(false)
                            {
                                toggle_fullscreen(
                                    window,
                                    &mut is_fullscreen,
                                    &mut saved_window_size,
                                    &mut saved_window_position,
                                );
                                last_click = None;
                                window.request_redraw();
                            } else {
                                last_click = Some(now);
                            }
                        }
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    logical_key: Key::Named(NamedKey::Escape),
                                    state: ElementState::Pressed,
                                    ..
                                },
                            ..
                        } => {
                            if is_fullscreen {
                                exit_fullscreen(
                                    window,
                                    &mut is_fullscreen,
                                    &mut saved_window_size,
                                    &mut saved_window_position,
                                );
                                if let Some(size) = latest_capture_size {
                                    gpu.update_transform(size);
                                }
                                window.request_redraw();
                            } else {
                                stop_capture(&capture_control);
                                elwt.exit();
                            }
                        }
                        WindowEvent::Resized(size) => {
                            if size.width == 0 || size.height == 0 {
                                return;
                            }
                            gpu.resize(size);
                            if let Some(size) = latest_capture_size {
                                gpu.update_transform(size);
                            }
                            window.request_redraw();
                        }
                        WindowEvent::ScaleFactorChanged {
                            mut inner_size_writer,
                            ..
                        } => {
                            if let Err(err) =
                                inner_size_writer.request_inner_size(window.inner_size())
                            {
                                eprintln!("unable to commit new size: {err}");
                            }
                            let new_size = window.inner_size();
                            if new_size.width == 0 || new_size.height == 0 {
                                return;
                            }
                            gpu.resize(new_size);
                            if let Some(size) = latest_capture_size {
                                gpu.update_transform(size);
                            }
                            window.request_redraw();
                        }
                        WindowEvent::RedrawRequested => {
                            if let Some(packet) = latest_frame(&shared_frame) {
                                if gpu
                                    .upload_frame(
                                        packet.pixels.as_ref(),
                                        (packet.width, packet.height),
                                    )
                                    .is_ok()
                                {
                                    latest_capture_size = Some((packet.width, packet.height));
                                }
                            }

                            match gpu.render() {
                                Ok(()) => {}
                                Err(wgpu::SurfaceError::Lost) => {
                                    gpu.resize(window.inner_size());
                                }
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    eprintln!("GPU out of memory");
                                    stop_capture(&capture_control);
                                    elwt.exit();
                                }
                                Err(wgpu::SurfaceError::Outdated) => {
                                    window.request_redraw();
                                }
                                Err(wgpu::SurfaceError::Timeout) => {
                                    eprintln!("Surface timeout");
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Event::UserEvent(UserEvent::FrameReady) => {
                    window.request_redraw();
                }
                Event::UserEvent(UserEvent::CaptureClosed) => {
                    stop_capture(&capture_control);
                    elwt.exit();
                }
                Event::AboutToWait => {
                    elwt.set_control_flow(ControlFlow::Wait);
                }
                _ => {}
            }
        })
        .map_err(|err| anyhow!(err))
}

fn minimum_update_interval(fps: u32) -> MinimumUpdateIntervalSettings {
    let clamped = fps.clamp(1, 240);
    MinimumUpdateIntervalSettings::Custom(Duration::from_secs_f64(1.0 / clamped as f64))
}

fn latest_frame(shared: &SharedFrame) -> Option<FramePacket> {
    shared.lock().unwrap().clone()
}

fn stop_capture(control: &Arc<Mutex<Option<CaptureControl<FrameDispatcher, anyhow::Error>>>>) {
    if let Some(ctrl) = control.lock().unwrap().take() {
        if let Err(err) = ctrl.stop() {
            eprintln!("failed to stop capture thread: {err:?}");
        }
    }
}

fn toggle_fullscreen(
    window: &Window,
    is_fullscreen: &mut bool,
    saved_size: &mut Option<PhysicalSize<u32>>,
    saved_pos: &mut Option<PhysicalPosition<i32>>,
) {
    if *is_fullscreen {
        exit_fullscreen(window, is_fullscreen, saved_size, saved_pos);
    } else {
        enter_fullscreen(window, is_fullscreen, saved_size, saved_pos);
    }
}

fn enter_fullscreen(
    window: &Window,
    is_fullscreen: &mut bool,
    saved_size: &mut Option<PhysicalSize<u32>>,
    saved_pos: &mut Option<PhysicalPosition<i32>>,
) {
    *saved_size = Some(window.inner_size());
    *saved_pos = window.outer_position().ok();
    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
    *is_fullscreen = true;
}

fn exit_fullscreen(
    window: &Window,
    is_fullscreen: &mut bool,
    saved_size: &mut Option<PhysicalSize<u32>>,
    saved_pos: &mut Option<PhysicalPosition<i32>>,
) {
    window.set_fullscreen(None);
    if let Some(pos) = saved_pos.take() {
        window.set_outer_position(pos);
    }
    if let Some(size) = saved_size.take() {
        let _ = window.request_inner_size(size);
    }
    *is_fullscreen = false;
}

fn resolve_display_index(explicit: Option<usize>, total: usize) -> Result<usize> {
    if let Some(idx) = explicit {
        if idx >= total {
            bail!("display {idx} is out of range (have {total})");
        }
        return Ok(idx);
    }

    prompt_for_index(total)
}

fn prompt_for_index(total: usize) -> Result<usize> {
    let mut input = String::new();
    loop {
        print!("Select a display [0-{}]: ", total.saturating_sub(1));
        io::stdout().flush().ok();
        input.clear();
        io::stdin()
            .read_line(&mut input)
            .context("failed to read selection")?;
        match input.trim().parse::<usize>() {
            Ok(idx) if idx < total => return Ok(idx),
            _ => println!(
                "Please enter a number between 0 and {}",
                total.saturating_sub(1)
            ),
        }
    }
}

struct TextureResources {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

struct GpuState<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    texture: Option<TextureResources>,
    capture_size: Option<(u32, u32)>,
}

impl<'window> GpuState<'window> {
    async fn new(window: &'window Window) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });

        let surface = instance
            .create_surface(window)
            .context("failed to create wgpu surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("no compatible GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("screenview-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screenview-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("transform-buffer"),
            contents: bytemuck::bytes_of(&TransformUniform::identity()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("uniform-layout"),
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform-bind-group"),
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("capture-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("screenview-pipeline-layout"),
            bind_group_layouts: &[&uniform_layout, &texture_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex-buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index-buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screenview-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            texture_layout,
            sampler,
            vertex_buffer,
            index_buffer,
            num_indices: INDICES.len() as u32,
            texture: None,
            capture_size: None,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        if let Some(size) = self.capture_size {
            self.update_transform(size);
        }
    }

    fn upload_frame(&mut self, data: &[u8], size: (u32, u32)) -> Result<()> {
        if self.texture.as_ref().map(|t| t.size) != Some(size) {
            self.texture = Some(self.create_texture(size));
        }

        if let Some(texture) = &self.texture {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(size.0 * 4),
                    rows_per_image: Some(size.1),
                },
                wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
            );
            self.capture_size = Some(size);
            self.update_transform(size);
        }

        Ok(())
    }

    fn update_transform(&self, capture: (u32, u32)) {
        let scale = aspect_scale(
            capture.0 as f32,
            capture.1 as f32,
            self.config.width as f32,
            self.config.height as f32,
        );
        let uniform = TransformUniform {
            scale,
            _pad: [0.0; 2],
        };
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    fn create_texture(&self, size: (u32, u32)) -> TextureResources {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture-texture"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("capture-bind-group"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        TextureResources {
            texture,
            bind_group,
            size,
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screenview-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if let Some(texture) = &self.texture {
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_bind_group(1, &texture.bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        drop(view);
        Ok(())
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct TransformUniform {
    scale: [f32; 2],
    _pad: [f32; 2],
}

impl TransformUniform {
    const fn identity() -> Self {
        Self {
            scale: [1.0, 1.0],
            _pad: [0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }
    }
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2];

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, 1.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
    Vertex {
        position: [1.0, -1.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0],
        uv: [0.0, 1.0],
    },
];

const INDICES: &[u16] = &[0, 1, 2, 0, 2, 3];

fn aspect_scale(capture_w: f32, capture_h: f32, surface_w: f32, surface_h: f32) -> [f32; 2] {
    if capture_w == 0.0 || capture_h == 0.0 || surface_w == 0.0 || surface_h == 0.0 {
        return [1.0, 1.0];
    }
    let capture_aspect = capture_w / capture_h;
    let surface_aspect = surface_w / surface_h;
    if surface_aspect > capture_aspect {
        [capture_aspect / surface_aspect, 1.0]
    } else {
        [1.0, surface_aspect / capture_aspect]
    }
}
