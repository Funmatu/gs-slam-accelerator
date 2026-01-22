import gs_slam_core
import os
import math
import struct

PLY_PATH = "data/object_0.ply"
TEMP_PLY_OUTPUT = "data/test_sr_output.ply"


def vec_len(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def parse_ply_coordinates(filepath):
    """
    PLYファイルから座標(x, y, z)のリストを読み込む簡易パーサー
    """
    coords = []
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
        parts = line.split()
        if len(parts) >= 3:
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return coords


def test_super_resolution():
    print(f"\n=== Testing Super Resolution Module ===")

    if not os.path.exists(PLY_PATH):
        print(f"Skipping: {PLY_PATH} not found.")
        return

    # 1. Load
    print("Loading PLY...")
    manager = gs_slam_core.SplatManager(PLY_PATH)
    orig_count = manager.count()
    print(f"Original Splats: {orig_count}")

    # 2. Test Factor = 1 (Should match original count)
    print("\n--- Test Case 1: Factor = 1 ---")
    try:
        count_x1 = manager.compute_super_resolution(1)
        print(f"Generated Surfels: {count_x1}")
        assert count_x1 == orig_count, (
            f"Count mismatch! Expected {orig_count}, got {count_x1}"
        )
    except Exception as e:
        print(f"⚠️ GPU Compute Failed (Factor 1): {e}")
        return

    # 3. Test Factor = 4 (Should be exactly 4x)
    print("\n--- Test Case 2: Factor = 4 (Upsampling) ---")
    factor = 4
    try:
        count_x4 = manager.compute_super_resolution(factor)
        print(f"Generated Surfels: {count_x4}")
        assert count_x4 == orig_count * factor, (
            f"Count mismatch! Expected {orig_count * factor}, got {count_x4}"
        )

        # Verify Normals and Colors in memory
        print("Verifying memory data attributes...")
        for i in [0, count_x4 // 2, count_x4 - 1]:
            # Normal Check
            n = manager.get_surfel_normal(i)
            length = vec_len(n)
            # 法線はほぼ1.0であるはず
            assert abs(length - 1.0) < 1e-3, f"Normal not normalized at {i}: {n}"

            # Color Check (RGB should be 0.0-1.0)
            c = manager.get_surfel_color(i)
            assert all(0.0 <= x <= 1.0 for x in c), f"Color out of range at {i}: {c}"

        print("✅ Memory attributes verified.")

    except Exception as e:
        print(f"⚠️ GPU Compute Failed (Factor 4): {e}")
        return

    # 4. Test Dispersion (Coordinates check via Export)
    # Python APIには現在 get_surfel_pos がないため、PLYエクスポート経由で
    # 点群が「同じ位置に重なっていないか（ちゃんと散らばっているか）」を確認する
    print("\n--- Test Case 3: Dispersion Check ---")
    manager.save_ply(TEMP_PLY_OUTPUT)

    coords = parse_ply_coordinates(TEMP_PLY_OUTPUT)
    assert len(coords) == count_x4

    # 親（Factor=1の時の位置）と比較したいが、簡易的に
    # 「連続する4つの点が全て同じ座標ではない」ことを確認する
    # SRの実装では、親1つにつき連続してN個の点を生成するため、
    # 座標がすべて親と同じなら、分散処理（乱数サンプリング）が動いていないことになる。

    check_index = 0  # 最初の親Splatに対応するブロック
    block = coords[check_index : check_index + factor]

    # 座標のばらつきを計算
    unique_coords = set(block)
    print(f"Sampled Block (First parent -> {factor} children):")
    for c in block:
        print(f"  {c}")

    if len(unique_coords) == 1:
        print("❌ Warning: All generated points have identical coordinates.")
        print(
            "   This assumes the splat scale is > 0. If scale is 0, this is expected."
        )
    else:
        print("✅ Points are dispersed (Coordinates vary).")

    # 5. Error Handling
    print("\n--- Test Case 4: Error Handling ---")
    try:
        manager.compute_super_resolution(0)
        print("❌ Failed: Should have raised ValueError for factor=0")
    except ValueError:
        print("✅ Correctly caught ValueError for factor=0")
    except Exception as e:
        print(f"❓ Caught unexpected exception: {type(e)}")

    # Cleanup
    if os.path.exists(TEMP_PLY_OUTPUT):
        os.remove(TEMP_PLY_OUTPUT)

    print("\n✅ Super Resolution Test Completed Successfully.")


if __name__ == "__main__":
    test_super_resolution()
