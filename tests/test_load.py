import gs_slam_core
import numpy as np

ply_path = "data/object_0.ply"
print(f"Loading {ply_path}...")

try:
    manager = gs_slam_core.SplatManager(ply_path)
    count = manager.count()
    print(f"Successfully loaded {count} splats.")

    # 最初のデータの検証 (インデックス0)
    pos = manager.get_splat_pos(0)
    rot = manager.get_splat_rot(0)
    sh = manager.get_splat_sh(0)

    print("-" * 40)
    print(f"Index 0 Check:")
    print(f"  Pos: {pos}")
    print(f"  Rot: {rot}")
    print(f"  SH:  {sh}")
    print("-" * 40)

    # データがゼロでないことを確認
    assert count > 0, "No splats loaded!"
    assert len(pos) == 3, "Position dimension mismatch"
    assert len(rot) == 4, "Rotation dimension mismatch"

    print("✅ Load Test Passed")

except Exception as e:
    print(f"❌ Load Test Failed: {e}")
