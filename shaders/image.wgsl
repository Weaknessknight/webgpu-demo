struct Params {
  width: u32,
  height: u32,
  algorithm: u32,
  radius: u32,
};

@group(0) @binding(0)
var<storage, read> inputPixels: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> outputPixels: array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> params: Params;

fn clampCoord(value: i32, limit: u32) -> u32 {
  return u32(clamp(value, 0, i32(limit) - 1));
}

fn luminance(x: i32, y: i32) -> f32 {
  let sx = clampCoord(x, params.width);
  let sy = clampCoord(y, params.height);
  let pixel = inputPixels[sy * params.width + sx];
  return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) {
    return;
  }

  let index = y * params.width + x;
  if (params.algorithm == 0u) {
    let radius = i32(params.radius);
    var sum = vec3<f32>(0.0);
    var count = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
      for (var dx = -radius; dx <= radius; dx = dx + 1) {
        let sx = clampCoord(i32(x) + dx, params.width);
        let sy = clampCoord(i32(y) + dy, params.height);
        sum += inputPixels[sy * params.width + sx].rgb;
        count += 1.0;
      }
    }
    outputPixels[index] = vec4<f32>(sum / count, 255.0);
  } else {
    let ix = i32(x);
    let iy = i32(y);
    let tl = luminance(ix - 1, iy - 1);
    let tc = luminance(ix, iy - 1);
    let tr = luminance(ix + 1, iy - 1);
    let ml = luminance(ix - 1, iy);
    let mr = luminance(ix + 1, iy);
    let bl = luminance(ix - 1, iy + 1);
    let bc = luminance(ix, iy + 1);
    let br = luminance(ix + 1, iy + 1);
    let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    let edge = min(255.0, sqrt(gx * gx + gy * gy));
    outputPixels[index] = vec4<f32>(edge, edge, edge, 255.0);
  }
}
