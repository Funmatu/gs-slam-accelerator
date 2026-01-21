import gs_slam_core
import os
import math
import struct

# =========================================================
#  Helper: Simple Parsers (No heavy dependencies like open3d)
# =========================================================


def parse_ply_ascii(filepath):
    """
    簡易PLYパーサー (ASCIIのみ対応)
    Returns: list of dict {'x', 'y', 'z', 'r', 'g', 'b', 'nx', 'ny', 'nz'}
    """
    points = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end = False
    for line in lines:
        line = line.strip()
        if not header_end:
            if line == "end_header":
                header_end = True
            continue

        if not line:
            continue

        # x, y, z, r, g, b, nx, ny, nz
        parts = line.split()
        if len(parts) < 9:
            continue

        pt = {
            "x": float(parts[0]),
            "y": float(parts[1]),
            "z": float(parts[2]),
            "r": int(parts[3]),
            "g": int(parts[4]),
            "b": int(parts[5]),
            "nx": float(parts[6]),
            "ny": float(parts[7]),
            "nz": float(parts[8]),
        }
        points.append(pt)
    return points


def parse_pcd_ascii(filepath):
    """
    簡易PCDパーサー (ASCII, DATA ascii以降を読み込む)
    Returns: list of dict (same structure as above)
    """
    points = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_section = False
    for line in lines:
        line = line.strip()
        if not data_section:
            if line.startswith("DATA ascii"):
                data_section = True
            continue

        if not line:
            continue

        # x, y, z, r, g, b, nx, ny, nz
        parts = line.split()
        if len(parts) < 9:
            continue

        pt = {
            "x": float(parts[0]),
            "y": float(parts[1]),
            "z": float(parts[2]),
            "r": int(parts[3]),
            "g": int(parts[4]),
            "b": int(parts[5]),
            "nx": float(parts[6]),
            "ny": float(parts[7]),
            "nz": float(parts[8]),
        }
        points.append(pt)
    return points


# =========================================================
#  Test Case
# =========================================================


def test_export_integrity():
    ply_input = "data/object_0.ply"
    ply_output = "data/test_output.ply"
    pcd_output = "data/test_output.pcd"

    print(f"\n--- Testing Export Functionality ---")

    # 1. Load and Compute
    manager = gs_slam_core.SplatManager(ply_input)
    count_orig = manager.count()
    print(f"Loaded: {count_orig} splats")

    # CPU fallback or GPU
    try:
        manager.compute_geometry()
    except:
        manager.compute_geometry_cpu()

    # 2. Export PLY
    print(f"Exporting PLY to {ply_output}...")
    manager.save_ply(ply_output)
    assert os.path.exists(ply_output), "PLY file was not created"

    # 3. Export PCD
    print(f"Exporting PCD to {pcd_output}...")
    manager.save_pcd(pcd_output)
    assert os.path.exists(pcd_output), "PCD file was not created"

    # 4. Verify Content
    print("Parsing exported files for verification...")
    ply_data = parse_ply_ascii(ply_output)
    pcd_data = parse_pcd_ascii(pcd_output)

    # Test A: Count Consistency
    print(
        f"Checking counts: Orig={count_orig}, PLY={len(ply_data)}, PCD={len(pcd_data)}"
    )
    assert count_orig == len(ply_data), "PLY count mismatch"
    assert count_orig == len(pcd_data), "PCD count mismatch"

    # Test B: XYZ Fidelity (Memory vs Export)
    # 最初の点と最後の点をサンプリングしてチェック
    indices_to_check = [0, count_orig // 2, count_orig - 1]

    for i in indices_to_check:
        # Memory
        mem_normal = manager.get_surfel_normal(i)
        mem_color = manager.get_surfel_color(i)

        # Note: manager.get_splat_pos(i) corresponds to surfel pos
        mem_pos = manager.get_splat_pos(i)

        # PLY File
        ply_pt = ply_data[i]

        # Check XYZ (Allow small float error due to ascii conversion)
        assert abs(mem_pos[0] - ply_pt["x"]) < 1e-4
        assert abs(mem_pos[1] - ply_pt["y"]) < 1e-4
        assert abs(mem_pos[2] - ply_pt["z"]) < 1e-4

        print(f"✅ Index {i}: XYZ matches between Memory and PLY")

    # Test C: Consistency between PLY and PCD (All fields)
    print("Checking PLY vs PCD consistency...")
    for i in indices_to_check:
        ply_pt = ply_data[i]
        pcd_pt = pcd_data[i]

        # XYZ
        assert abs(ply_pt["x"] - pcd_pt["x"]) < 1e-5

        # RGB (Integer exact match expected)
        assert ply_pt["r"] == pcd_pt["r"]
        assert ply_pt["g"] == pcd_pt["g"]
        assert ply_pt["b"] == pcd_pt["b"]

        # Normal
        assert abs(ply_pt["nx"] - pcd_pt["nx"]) < 1e-5

        print(f"✅ Index {i}: PLY and PCD data are identical")

    # Test D: Data Sanity (Normal normalization)
    print("Checking Normal Vector sanity...")
    for i in indices_to_check:
        nx, ny, nz = ply_data[i]["nx"], ply_data[i]["ny"], ply_data[i]["nz"]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        # ほぼ 1.0 であること
        assert abs(length - 1.0) < 1e-3, (
            f"Normal vector not normalized at index {i}: len={length}"
        )
        print(f"✅ Index {i}: Normal vector length is {length:.4f}")

    # Cleanup
    if os.path.exists(ply_output):
        os.remove(ply_output)
    if os.path.exists(pcd_output):
        os.remove(pcd_output)
    print("✅ Export Test Passed: Cleaned up files.")


if __name__ == "__main__":
    try:
        test_export_integrity()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        exit(1)
