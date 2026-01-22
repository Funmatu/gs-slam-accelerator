// src/sr.wgsl

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

// 決定論的なパターン生成のためのヘルパー
// Golden Angleに基づく螺旋配置は、円盤を均一に埋めるのに適している
fn concentric_sample(index: u32, total: u32) -> vec2<f32> {
    if (index == 0u) { return vec2<f32>(0.0, 0.0); }
    
    // Golden Angle Spiral Distribution
    // これにより、乱数を使わずに均一かつ密なサンプリングが可能
    let theta = f32(index) * 2.3999632; // 2.399... = Golden Angle (radians)
    let r = sqrt(f32(index) / f32(total)); // Area preserving radius
    
    // r は 0.0~1.0。これをガウシアンの有効範囲（通常3シグマだが、SLAM用に1.0シグマに留める）にマップ
    // 中心付近を重視するため、少し内側に寄せる (0.8倍)
    let r_scaled = r * 0.8; 

    return vec2<f32>(r_scaled * cos(theta), r_scaled * sin(theta));
}

@compute @workgroup_size(64)
fn compute_sr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_surfels = arrayLength(&input_splats) * params.factor;
    
    if (idx >= total_surfels) { return; }

    let parent_idx = idx / params.factor;
    let child_idx = idx % params.factor; // 0 to factor-1
    let splat = input_splats[parent_idx];

    // =========================================================
    // 1. Strict Filtering for SLAM (精度向上のための選別)
    // =========================================================
    
    let s = abs(splat.scale);
    let max_s = max(s.x, max(s.y, s.z));
    let min_s = min(s.x, min(s.y, s.z));
    
    // アスペクト比チェック: 
    // 球体に近い(min/max > 0.3)ものは「面」ではないため、SLAMの幾何拘束に使えない。
    // オパシティチェック:
    // 薄いものはノイズ。
    var valid = true;
    if (splat.opacity < 0.3) { valid = false; }
    if (min_s / max_s > 0.4) { valid = false; } 

    if (!valid) {
        // 無効なSplatは「退化」させる（座標0またはNaNを入れる）
        // ここでは安全のため親と同じ位置（オフセット0）にし、法線0としてマーキングする手もあるが、
        // 単に親の位置に置いておく。
        var out_invalid: Surfel;
        out_invalid.pos = splat.pos;
        out_invalid.color = vec3<f32>(0.0); // 黒くして目立たなくするか
        out_invalid.normal = vec3<f32>(0.0);
        output_surfels[idx] = out_invalid;
        return;
    }

    // =========================================================
    // 2. Geometry Calculation
    // =========================================================

    let q = normalize(splat.rot);
    let q_fixed = vec4<f32>(q.y, q.z, q.w, q.x);
    let R = quat_to_mat3(q_fixed);

    // 法線と接平面軸の特定
    var local_n = vec3<f32>(0.0, 0.0, 1.0);
    var tangent_u = vec3<f32>(1.0, 0.0, 0.0);
    var tangent_v = vec3<f32>(0.0, 1.0, 0.0);
    var scale_u = s.x;
    var scale_v = s.y;

    if (s.x < s.y && s.x < s.z) { 
        local_n = vec3<f32>(1.0, 0.0, 0.0); 
        tangent_u = vec3<f32>(0.0, 1.0, 0.0); scale_u = s.y;
        tangent_v = vec3<f32>(0.0, 0.0, 1.0); scale_v = s.z;
    } else if (s.y < s.z) { 
        local_n = vec3<f32>(0.0, 1.0, 0.0); 
        tangent_u = vec3<f32>(1.0, 0.0, 0.0); scale_u = s.x;
        tangent_v = vec3<f32>(0.0, 0.0, 1.0); scale_v = s.z;
    }

    let normal = normalize(R * local_n);

    // =========================================================
    // 3. Deterministic Planar Sampling (決定論的平面サンプリング)
    // =========================================================
    
    var offset_local = vec3<f32>(0.0);

    if (params.factor > 1u) {
        // 乱数ではなく、黄金角(Golden Angle)螺旋で配置
        // これにより、常に均一で「平ら」な円盤が得られる
        let sample_pt = concentric_sample(child_idx, params.factor); // vec2 (-1.0 ~ 1.0)
        
        // 接平面上でのみ展開 (法線方向への移動はゼロ！)
        // scale_u, scale_v は標準偏差(1 sigma)
        offset_local = (tangent_u * sample_pt.x * scale_u) + (tangent_v * sample_pt.y * scale_v);
    }

    let pos_world = splat.pos + R * offset_local;

    // SH to RGB
    let C0 = 0.2820947917;
    var rgb = 0.5 + C0 * splat.sh_dc;
    rgb = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));

    var out: Surfel;
    out.pos = pos_world;
    out.color = rgb;
    out.normal = normal;
    
    out._pad0 = 0.0; out._pad1 = 0.0; out._pad2 = 0.0;

    output_surfels[idx] = out;
}