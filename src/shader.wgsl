// src/shader.wgsl

struct GaussianSplat {
    pos: vec3<f32>,
    opacity: f32,
    scale: vec3<f32>,
    _pad1: f32,
    rot: vec4<f32>, // Quaternion (r, i, j, k)
    sh_dc: vec3<f32>,
    _pad2: f32,
}

struct Surfel {
    pos: vec3<f32>,      // 12 bytes
    _pad0: f32,          // 4 bytes
    normal: vec3<f32>,   // 12 bytes
    _pad1: f32,          // 4 bytes
    cov_row0: vec3<f32>, // 12 bytes (共分散行列 1行目)
    _pad2: f32,
    cov_row1: vec3<f32>, // 12 bytes (共分散行列 2行目)
    _pad3: f32,
    cov_row2: vec3<f32>, // 12 bytes (共分散行列 3行目)
    _pad4: f32,
}

@group(0) @binding(0)
var<storage, read> input_splats: array<GaussianSplat>;

@group(0) @binding(1)
var<storage, read_write> output_surfels: array<Surfel>;

// クォータニオンから回転行列への変換
fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let r = q.x; let x = q.y; let y = q.z; let z = q.w;
    
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y + z*z), 2.0*(x*y + r*z),       2.0*(x*z - r*y)),
        vec3<f32>(2.0*(x*y - r*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + r*x)),
        vec3<f32>(2.0*(x*z + r*y),       2.0*(y*z - r*x),       1.0 - 2.0*(x*x + y*y))
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input_splats)) {
        return;
    }

    let splat = input_splats[idx];

    // 1. パラメータの正規化・変換
    // 3DGSのスケールは exp() されていることが多いが、生データが exp 済みか確認が必要。
    // 今回はバイナリ値が -6.0 程度だったため、exp() が必要と推測される。
    let s = exp(splat.scale); 
    
    // クォータニオンの正規化
    let q = normalize(splat.rot);
    let R = quat_to_mat3(q);
    
    // スケール行列 S (対角成分のみ)
    let S = mat3x3<f32>(
        vec3<f32>(s.x, 0.0, 0.0),
        vec3<f32>(0.0, s.y, 0.0),
        vec3<f32>(0.0, 0.0, s.z)
    );

    // 2. 法線の抽出 (最小スケールの軸を選択)
    // Rの列ベクトルはそれぞれローカル軸 (x, y, z) のワールド方向を表す
    var normal = vec3<f32>(0.0, 0.0, 0.0);
    
    if (s.x <= s.y && s.x <= s.z) {
        normal = R[0]; // Rの1列目
    } else if (s.y <= s.x && s.y <= s.z) {
        normal = R[1]; // Rの2列目
    } else {
        normal = R[2]; // Rの3列目
    }

    // 3. 共分散行列の計算: Sigma = R * S * S^T * R^T
    // M = R * S
    let M = R * S; 
    let Sigma = M * transpose(M); // M * M^T

    // 4. 結果の書き込み
    output_surfels[idx].pos = splat.pos;
    output_surfels[idx].normal = normal;
    output_surfels[idx].cov_row0 = Sigma[0];
    output_surfels[idx].cov_row1 = Sigma[1];
    output_surfels[idx].cov_row2 = Sigma[2];
}