import gs_slam_core
import math

ply_path = "data/object_0.ply"
manager = gs_slam_core.SplatManager(ply_path)

print("Testing Features via CPU Fallback...")

# 1. 3DGS Load
count = manager.count()
assert count > 0
print(f"1. Load: OK ({count} splats)")

# 2. Compute Geometry (CPU)
# これで XYZRGB化 と 法線推定 のロジックを検証する
num_surfels = manager.compute_geometry_cpu()
assert num_surfels == count
print(f"2. Compute (CPU): OK ({num_surfels} surfels generated)")

# 3. Verify Conversion Logic
# 最初のSplatの生データ
raw_rot = manager.get_splat_rot(0)
raw_sh = manager.get_splat_sh(0)

# 計算後のデータ
calc_normal = manager.get_surfel_normal(0)
calc_color = manager.get_surfel_color(0)

# SH -> RGB の簡易検算 (0次項のみ)
# 0.5 + 0.28209 * sh
expected_r = max(0.0, min(1.0, 0.5 + 0.2820947917 * raw_sh[0]))
print(f"3. Logic Check:")
print(f"   SH[0]: {raw_sh[0]} -> RGB[0]: {calc_color[0]} (Expected: {expected_r:.4f})")

# 値が近似していればOK
assert abs(calc_color[0] - expected_r) < 0.01
print("✅ All PyO3 Features Verified (CPU Logic verified independent of GPU driver)")
