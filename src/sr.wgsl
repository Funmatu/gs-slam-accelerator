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

// --- Helpers ---

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

// Sigmoid function for opacity (if stored as logit)
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn concentric_sample(index: u32, total: u32) -> vec2<f32> {
    if (index == 0u) { return vec2<f32>(0.0, 0.0); }
    let theta = f32(index) * 2.3999632; // Golden Angle
    let r = sqrt(f32(index) / f32(total));
    // 0.8倍することで、ガウシアンの裾野(薄い部分)を避けて中心寄りに配置する
    return vec2<f32>(r * 0.8 * cos(theta), r * 0.8 * sin(theta));
}

@compute @workgroup_size(64)
fn compute_sr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_surfels = arrayLength(&input_splats) * params.factor;
    
    if (idx >= total_surfels) { return; }

    let parent_idx = idx / params.factor;
    let child_idx = idx % params.factor;
    let splat = input_splats[parent_idx];

    // =========================================================
    // 1. Scale & Opacity Correction (重要！)
    // =========================================================
    
    // 【修正】スケールは対数で保存されているため、exp() で実数に戻す
    let s = exp(splat.scale);
    
    // 【修正】オパシティもSigmoidを通す（一般的なPLYの場合）
    // もしすでに0-1の範囲ならこの行は不要だが、通しても害は少ない
    let opacity = sigmoid(splat.opacity);

    // =========================================================
    // 2. Filtering (SLAM用にノイズ除去)
    // =========================================================

    let max_s = max(s.x, max(s.y, s.z));
    let min_s = min(s.x, min(s.y, s.z));
    
    var valid = true;
    
    // オパシティ閾値 (薄い霧を除去)
    if (opacity < 0.3) { valid = false; }
    
    // アスペクト比閾値 (球体に近い形状を除去し、平らな面だけ残す)
    // 0.6以上の比率がある＝丸っこい＝壁ではない可能性
    if (min_s / max_s > 0.6) { valid = false; } 

    // スケールが大きすぎるものを除去（背景の巨大なビルボード等）
    // シーンによりますが、極端に大きい板は精度を下げる要因
    // if (max_s > 10.0) { valid = false; }

    if (!valid) {
        // 無効な点は描画しない（縮退させる）
        var out_invalid: Surfel;
        out_invalid.pos = vec3<f32>(0.0);
        out_invalid.color = vec3<f32>(0.0);
        out_invalid.normal = vec3<f32>(0.0);
        // padding
        out_invalid._pad0 = 0.0; out_invalid._pad1 = 0.0; out_invalid._pad2 = 0.0;
        output_surfels[idx] = out_invalid;
        return;
    }

    // =========================================================
    // 3. Geometry Calculation
    // =========================================================

    let q = normalize(splat.rot);
    let q_fixed = vec4<f32>(q.y, q.z, q.w, q.x);
    let R = quat_to_mat3(q_fixed);

    // 法線方向(最も薄い軸)を特定
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
    // 4. Deterministic Planar Sampling
    // =========================================================
    
    var offset_local = vec3<f32>(0.0);

    if (params.factor > 1u) {
        let sample_pt = concentric_sample(child_idx, params.factor);
        
        // 接平面上のみに展開。法線方向への移動はゼロ！
        // これで「表面に張り付いた」点群になる
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
