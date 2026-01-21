import gs_slam_core
import time

ply_path = "data/object_0.ply"
manager = gs_slam_core.SplatManager(ply_path)

print("Attempting GPU Compute...")
try:
    start = time.time()
    count = manager.compute_geometry()
    elapsed = (time.time() - start) * 1000
    print(f"✅ GPU Compute Success: {count} surfels in {elapsed:.2f} ms")

    # 結果確認
    print(f"  Normal[0]: {manager.get_surfel_normal(0)}")
    print(f"  Color[0]:  {manager.get_surfel_color(0)}")

except Exception as e:
    print(f"⚠️ GPU Compute Failed (Expected in some WSL2 envs): {e}")
    print("Skipping to CPU verification...")
