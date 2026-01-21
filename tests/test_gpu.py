import gs_slam_core
import time

ply_path = "data/object_0.ply"  # パスは適宜変更

print("1. Loading PLY...")
manager = gs_slam_core.SplatManager(ply_path)
print(f"Loaded {manager.count()} splats.")

print("2. Running GPU Compute...")
start = time.time()

# ここでGPUが火を吹きます
count = manager.compute_geometry()

elapsed = (time.time() - start) * 1000
print(f"Computed {count} surfels in {elapsed:.2f} ms")

print("3. Result Validation")
print(manager.debug_first_surfel())
