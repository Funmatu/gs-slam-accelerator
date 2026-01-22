// ============================================================================
//  Super Resolution Compute Shader
// ============================================================================

struct GaussianSplat {
    pos: vec3<f32>,
    opacity: f32,
    scale: vec3<f32>,
    _pad1: f32,
    rot: vec4<f32>,
    sh_dc: vec3<f32>,
    _pad2: f32,
};

struct Surfel {
    pos: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    _pad1: f32,
    normal: vec3<f32>,
    _pad2: f32,
};

struct Params {
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> input_splats : array<GaussianSplat>;
@group(0) @binding(1) var<storage, read_write> output_surfels : array<Surfel>;
@group(0) @binding(2) var<uniform> params : Params;

// --- Math Helpers ---

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

// PCG Hash for fast, high-quality random numbers
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967296.0;
}

// Box-Muller transform for Normal Distribution
fn rand_normal(seed: u32) -> vec2<f32> {
    let u1 = max(rand_float(seed), 1e-7); // avoid log(0)
    let u2 = rand_float(seed + 137u);
    
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.2831853 * u2;
    
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute @workgroup_size(64)
fn compute_sr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_surfels = arrayLength(&input_splats) * params.factor;
    
    if (idx >= total_surfels) { return; }

    // Identify Parent Splat
    let parent_idx = idx / params.factor;
    let splat = input_splats[parent_idx];

    // 1. Basic Geometry
    let q = normalize(splat.rot);
    let q_fixed = vec4<f32>(q.y, q.z, q.w, q.x); // w,x,y,z -> x,y,z,w mapping
    let R = quat_to_mat3(q_fixed);
    let s = abs(splat.scale);

    // 2. Determine Local Normal (Min Scale Axis) & Tangent Plane
    var local_n = vec3<f32>(0.0, 0.0, 1.0);
    var mask = vec3<f32>(1.0, 1.0, 0.0); // xy plane sampling by default
    
    if (s.x < s.y && s.x < s.z) { 
        local_n = vec3<f32>(1.0, 0.0, 0.0); 
        mask = vec3<f32>(0.0, 1.0, 1.0); // yz plane
    } else if (s.y < s.z) { 
        local_n = vec3<f32>(0.0, 1.0, 0.0); 
        mask = vec3<f32>(1.0, 0.0, 1.0); // xz plane
    }

    let normal = normalize(R * local_n);

    // 3. Sampling Logic
    // If factor == 1, just use center (no offset).
    // If factor > 1, sample from distribution.
    var offset_local = vec3<f32>(0.0);
    
    if (params.factor > 1u) {
        // Unique seed per output point
        let seed = idx * 19937u + params.factor; 
        let rnd = rand_normal(seed); // vec2 (standard normal)
        
        // Map random values to the 2 axes defined by 'mask'
        // We use 1.0 sigma. 
        // Note: 3DGS scale usually represents log-scale or direct covariance.
        // Assuming 'scale' is standard deviation (sigma) here.
        // We only perturb on the tangent plane (Disk Sampling).
        
        var r_idx = 0;
        if (mask.x > 0.5) { offset_local.x = rnd[r_idx] * s.x; r_idx++; }
        if (mask.y > 0.5) { offset_local.y = rnd[r_idx] * s.y; r_idx++; }
        if (mask.z > 0.5) { offset_local.z = rnd[r_idx] * s.z; }
    }

    let pos_world = splat.pos + R * offset_local;

    // 4. Color Logic (SH to RGB)
    let C0 = 0.2820947917;
    var rgb = 0.5 + C0 * splat.sh_dc;
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    // 5. Output
    var out: Surfel;
    out.pos = pos_world;
    out.color = rgb;
    out.normal = normal;
    
    out._pad0 = 0.0;
    out._pad1 = 0.0;
    out._pad2 = 0.0;

    output_surfels[idx] = out;
}
