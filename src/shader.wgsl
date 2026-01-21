struct GaussianSplat {
    pos: vec3<f32>,
    opacity: f32, // w component of vec4
    scale: vec3<f32>,
    _pad1: f32,   // w component of vec4
    rot: vec4<f32>,
    sh_dc: vec3<f32>,
    _pad2: f32,   // w component of vec4
};

struct Surfel {
    pos: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    _pad1: f32,
    normal: vec3<f32>,
    _pad2: f32,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad: f32,
    display_mode: u32,
    _pad3: vec3<u32>, // Pad to 16 bytes alignment
};

// --- Compute Shader ---

@group(0) @binding(0) var<storage, read> input_splats : array<GaussianSplat>;
@group(0) @binding(1) var<storage, read_write> output_surfels : array<Surfel>;

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x*x; let y2 = y*y; let z2 = z*z;
    let xy = x*y; let xz = x*z; let yz = y*z;
    let wx = w*x; let wy = w*y; let wz = w*z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y2 + z2), 2.0*(xy + wz),       2.0*(xz - wy)),
        vec3<f32>(2.0*(xy - wz),       1.0 - 2.0*(x2 + z2), 2.0*(yz + wx)),
        vec3<f32>(2.0*(xz + wy),       2.0*(yz - wx),       1.0 - 2.0*(x2 + y2))
    );
}

@compute @workgroup_size(64)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input_splats)) { return; }

    let splat = input_splats[idx];

    // SH (DC) to Color
    let C0 = 0.2820947917;
    var rgb = 0.5 + C0 * splat.sh_dc;
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    // Normal Estimation
    // rot is stored as [x, y, z, w] in Rust/Struct, so directly usable here if assumed same convention.
    // If PLY is [w, x, y, z], mapping would be needed. 
    // Usually PLY from 3DGS is [r, x, y, z] (w,x,y,z).
    // Rust loader just copies bytes.
    // Let's assume standard quaternion math works.
    
    // Normalization is key.
    let q = normalize(splat.rot); 
    // Note: If data is (w, x, y, z), q.x=w, q.y=x... which is mixed up for standard math (x,y,z,w).
    // Correct mapping for (w, x, y, z) input to (x, y, z, w) logic:
    let q_fixed = vec4<f32>(q.y, q.z, q.w, q.x); // x, y, z, w
    
    let R = quat_to_mat3(q_fixed);
    let s = abs(splat.scale);
    
    var local_n = vec3<f32>(0.0, 0.0, 1.0);
    if (s.x < s.y && s.x < s.z) { local_n = vec3<f32>(1.0, 0.0, 0.0); }
    else if (s.y < s.z) { local_n = vec3<f32>(0.0, 1.0, 0.0); }
    
    let normal = normalize(R * local_n);

    var out: Surfel;
    out.pos = splat.pos;
    out.color = rgb;
    out.normal = normal;
    
    // Explicit padding init (good practice)
    out._pad0 = 0.0;
    out._pad1 = 0.0;
    out._pad2 = 0.0;

    output_surfels[idx] = out;
}

// --- Render Shader ---

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.pos, 1.0);

    if (camera.display_mode == 0u) {
        out.color = in.color;
    } else {
        out.color = in.normal * 0.5 + 0.5;
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
