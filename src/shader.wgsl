struct Transform {
    scale: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> transform: Transform;

@group(1) @binding(0)
var capture_tex: texture_2d<f32>;

@group(1) @binding(1)
var capture_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    let scaled = position * transform.scale;
    out.position = vec4<f32>(scaled, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(capture_tex, capture_sampler, in.uv);
}
