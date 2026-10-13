"""
scripts/reshard_safetensors.py
Split a single huge model.safetensors into ~2 GB shards, reading with plain
file I/O (constant memory).

Why: on Windows, safetensors memory-maps the whole file copy-on-write, which
charges its full size against the commit limit. A 16 GB single-file checkpoint
then fails with "The paging file is too small" on a machine with a small
page file, while ~2 GB shards load fine one at a time.

Usage:
    python scripts/reshard_safetensors.py <model_dir> <out_dir> [--shard-gb 2]
"""

import argparse
import json
import os
import shutil
import struct

CHUNK = 64 * 1024 * 1024


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def write_shard(src, data_start: int, tensors: list[tuple[str, dict]], out_path: str):
    header, offset = {}, 0
    for name, info in tensors:
        begin, end = info["data_offsets"]
        header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)                      # header length must keep data 8-byte aligned
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for _, info in tensors:
            begin, end = info["data_offsets"]
            src.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                block = src.read(min(CHUNK, remaining))
                out.write(block)
                remaining -= len(block)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir")
    p.add_argument("out_dir")
    p.add_argument("--shard-gb", type=float, default=2.0)
    args = p.parse_args()

    src_path = os.path.join(args.model_dir, "model.safetensors")
    header, data_start = read_header(src_path)
    metadata = header.pop("__metadata__", None)
    tensors = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])

    limit = args.shard_gb * 1024 ** 3
    shards, current, size = [], [], 0
    for name, info in tensors:
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        if current and size + nbytes > limit:
            shards.append(current)
            current, size = [], 0
        current.append((name, info))
        size += nbytes
    shards.append(current)

    os.makedirs(args.out_dir, exist_ok=True)
    weight_map, total = {}, 0
    with open(src_path, "rb") as src:
        for i, shard in enumerate(shards, 1):
            fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            write_shard(src, data_start, shard, os.path.join(args.out_dir, fname))
            for name, info in shard:
                weight_map[name] = fname
                total += info["data_offsets"][1] - info["data_offsets"][0]
            print(f"{fname}: {len(shard)} tensors")

    with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total, **(metadata or {})}, "weight_map": weight_map}, f, indent=2)
    for name in os.listdir(args.model_dir):
        if name != "model.safetensors" and os.path.isfile(os.path.join(args.model_dir, name)):
            shutil.copy2(os.path.join(args.model_dir, name), os.path.join(args.out_dir, name))
    print(f"{len(shards)} shards, {total / 1024 ** 3:.1f} GB -> {args.out_dir}")


if __name__ == "__main__":
    main()
