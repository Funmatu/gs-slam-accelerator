import gs_slam_core
import time
import os

# Ensure you have a valid .ply file from a 3DGS training result
ply_file = "data/object_0.ply"

if not os.path.exists(ply_file):
    print(f"Please provide a valid path. File not found: {ply_file}")
    exit(1)

print(f"Loading {ply_file}...")
start_time = time.time()

# This triggers Rust mmap + parsing
manager = gs_slam_core.SplatManager(ply_file)

end_time = time.time()
elapsed = (end_time - start_time) * 1000

count = manager.count()
print(f"--------------------------------------------------")
print(f"Loaded {count:,} splats in {elapsed:.2f} ms")
print(f"Throughput: {count / (elapsed / 1000) / 1_000_000:.2f} M splats/sec")
print(f"First Splat Data: {manager.debug_first_splat()}")
print(f"--------------------------------------------------")
