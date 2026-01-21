import struct

filename = "data/object_0.ply"

with open(filename, "rb") as f:
    # 1. ヘッダーを読み飛ばす（end_headerを探す）
    content = b""
    while True:
        chunk = f.read(1)
        content += chunk
        if b"end_header" in content:
            # 改行コードの処理（\n または \r\n）
            char = f.read(1)
            if char == b"\r":
                f.read(1)  # \nをスキップ
            break

    print("--- Binary Header End ---")

    # 2. 最初の頂点のデータを読み込む
    # ヘッダーによると: x, y, z, nx, ny, nz, f_dc_0...rot_3 (合計17個のfloat)
    # float32 (4bytes) * 17 = 68 bytes

    data = f.read(68)
    if len(data) < 68:
        print("Error: Not enough data for one vertex")
        exit()

    # 3. float32 (Little Endian) として解釈
    # <f: Little Endian float
    values = struct.unpack("<17f", data)

    properties = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]

    print(f"{'Property':<10} | {'Value':<20}")
    print("-" * 35)
    for name, val in zip(properties, values):
        print(f"{name:<10} | {val:.6f}")
