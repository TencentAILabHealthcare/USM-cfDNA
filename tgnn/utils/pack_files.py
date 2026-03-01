import io
import struct
import zlib
import gzip
import lzma
import contextlib
from typing import Dict, Any


try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    zstd = None
    HAS_ZSTD = False



MAGIC = b"ASPK"
VERSION = 1

# Header: magic(4s) + version(u16) + index_offset(u64) + index_len(u64)
PACK_HDR = struct.Struct("<4sHQQ")

# Algo IDs
ALGO_NONE = 0
ALGO_ZSTD = 1
ALGO_GZIP = 2
ALGO_LZMA = 3

ALGO_MAP = {
    "none": ALGO_NONE,
    "zstd": ALGO_ZSTD,
    "gzip": ALGO_GZIP,
    "lzma": ALGO_LZMA,
}

# Index entry fixed part:
# algo(u8) + crc32(u32) + raw_len(u64) + comp_len(u64) + offset(u64) + name_len(u16)
ENTRY_FIXED = struct.Struct("<BIQQQH")


class LimitedReader(io.RawIOBase):
    """Wrap a file object and allow reading at most `limit` bytes."""

    def __init__(self, f, limit: int):
        self._f = f
        self._remain = limit

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if self._remain <= 0:
            return b""
        if n is None or n < 0 or n > self._remain:
            n = self._remain
        b = self._f.read(n)
        self._remain -= len(b)
        return b

    def readinto(self, b) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


@contextlib.contextmanager
def _compress_writer(fout, algo_id: int, level: int):
    """
    Yield a binary writer that compresses input into `fout`.
    Must NOT close `fout` when closing the writer.
    """
    if algo_id == ALGO_NONE:
        class _NoCloseWriter:
            def __init__(self, f): self._f = f
            def write(self, data: bytes): return self._f.write(data)
            def flush(self): return self._f.flush()
            def close(self): return

        w = _NoCloseWriter(fout)
        yield w
        return

    if algo_id == ALGO_GZIP:
        lvl = max(0, min(9, level))
        w = gzip.GzipFile(fileobj=fout, mode="wb", compresslevel=lvl)
        try:
            yield w
        finally:
            w.close()
        return

    if algo_id == ALGO_LZMA:
        lvl = max(0, min(9, level))
        w = lzma.LZMAFile(fout, mode="wb", preset=lvl)
        try:
            yield w
        finally:
            w.close()
        return

    if algo_id == ALGO_ZSTD:
        if not HAS_ZSTD:
            raise RuntimeError("algo='zstd' requires `pip install zstandard`")
        w = zstd.ZstdCompressor(level=level).stream_writer(fout)  # type: ignore
        try:
            yield w
        finally:
            w.close()
        return

    raise ValueError(f"unknown algo_id: {algo_id}")


def _decompress_reader(fin, algo_id: int):
    """Return a binary reader that yields decompressed bytes."""
    if algo_id == ALGO_NONE:
        return fin
    if algo_id == ALGO_GZIP:
        return gzip.GzipFile(fileobj=fin, mode="rb")
    if algo_id == ALGO_LZMA:
        return lzma.LZMAFile(fin, mode="rb")
    if algo_id == ALGO_ZSTD:
        if not HAS_ZSTD:
            raise RuntimeError("zstd in pack but `zstandard` is not installed")
        return zstd.ZstdDecompressor().stream_reader(fin)  # type: ignore
    raise ValueError(f"unknown algo_id: {algo_id}")


def _read_index(f):
    """
    Read header + index. Return:
    - entries: name -> dict(algo, crc32, raw_len, comp_len, offset)
    - data_start: offset where data section starts
    """
    head = f.read(PACK_HDR.size)
    if len(head) != PACK_HDR.size:
        raise ValueError("bad pack: too small")

    magic, ver, index_offset, index_len = PACK_HDR.unpack(head)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("not an ASPK pack or unsupported version")

    f.seek(index_offset)
    idx = f.read(index_len)
    if len(idx) != index_len:
        raise ValueError("bad pack: index truncated")

    mv = memoryview(idx)
    off = 0

    if len(mv) < 4:
        raise ValueError("bad pack: index too small")

    n = struct.unpack_from("<I", mv, off)[0]
    off += 4

    entries: Dict[str, Dict[str, int]] = {}
    for _ in range(n):
        if off + ENTRY_FIXED.size > len(mv):
            raise ValueError("bad pack: entry truncated")

        algo, crc32v, raw_len, comp_len, offset, name_len = ENTRY_FIXED.unpack_from(mv, off)
        off += ENTRY_FIXED.size

        if off + name_len > len(mv):
            raise ValueError("bad pack: name truncated")

        name = mv[off:off + name_len].tobytes().decode("utf-8")
        off += name_len

        entries[name] = {
            "algo": int(algo),
            "crc32": int(crc32v),
            "raw_len": int(raw_len),
            "comp_len": int(comp_len),
            "offset": int(offset),
        }

    data_start = PACK_HDR.size
    return entries, data_start



def pack_files(
    name_to_path: Dict[str, str],
    out_pack_path: str,
    algo: str = "gzip",
    level: int = 6,
    chunk_size: int = 1024 * 1024,
) -> None:
    """
    Pack multiple files into one pack.

    algo: "none" | "gzip" | "lzma" | "zstd"
    level: compression level (gzip/lzma: 0-9; zstd: typical 1-19+)
    """
    if algo not in ALGO_MAP:
        raise ValueError(f"algo must be one of {list(ALGO_MAP.keys())}")

    algo_id = ALGO_MAP[algo]

    entries = []

    with open(out_pack_path, "wb") as fout:
        # 1) write placeholder header (index_offset/index_len will be patched later)
        fout.write(PACK_HDR.pack(MAGIC, VERSION, 0, 0))
        data_start = PACK_HDR.size

        # 2) write data blobs sequentially, record metadata
        for name, path in name_to_path.items():
            if not isinstance(name, str) or not name:
                raise ValueError("resource name must be a non-empty str")

            name_b = name.encode("utf-8")
            if len(name_b) > 65535:
                raise ValueError(f"resource name too long: {name}")

            start_pos = fout.tell()
            offset = start_pos - data_start  # offset relative to data section start

            crc = 0
            raw_len = 0

            with open(path, "rb") as fin, _compress_writer(fout, algo_id, level) as w:
                while True:
                    chunk = fin.read(chunk_size)
                    if not chunk:
                        break
                    raw_len += len(chunk)
                    crc = zlib.crc32(chunk, crc)
                    w.write(chunk)

            end_pos = fout.tell()
            comp_len = end_pos - start_pos

            entries.append({
                "name": name,
                "name_b": name_b,
                "name_len": len(name_b),
                "algo": algo_id,
                "crc32": crc & 0xFFFFFFFF,
                "raw_len": raw_len,
                "comp_len": comp_len,
                "offset": offset,
            })

        # 3) write index at end
        index_offset = fout.tell()
        fout.write(struct.pack("<I", len(entries)))  # entry count

        for e in entries:
            fout.write(ENTRY_FIXED.pack(
                e["algo"],
                e["crc32"],
                e["raw_len"],
                e["comp_len"],
                e["offset"],
                e["name_len"],
            ))
            fout.write(e["name_b"])

        index_end = fout.tell()
        index_len = index_end - index_offset

        # 4) patch header
        fout.seek(0)
        fout.write(PACK_HDR.pack(MAGIC, VERSION, index_offset, index_len))


def list_resources(pack_path: str):
    """Return all resource names inside pack."""
    with open(pack_path, "rb") as f:
        entries, _ = _read_index(f)
    return sorted(entries.keys())


def read_resource_bytes(pack_path: str, name: str, verify_crc32: bool = True):
    """
    Read one resource as bytes.
    If verify_crc32=True, it checks CRC32 and raw_len (detect accidental corruption).
    """
    with open(pack_path, "rb") as f:
        entries, data_start = _read_index(f)
        if name not in entries:
            raise KeyError(f"resource not found: {name}")

        e = entries[name]
        f.seek(data_start + e["offset"])

        limited = LimitedReader(f, e["comp_len"])
        reader = _decompress_reader(limited, e["algo"])

        raw = reader.read()
        try:
            reader.close()
        except Exception:
            pass

        if len(raw) != e["raw_len"]:
            raise ValueError("length mismatch: pack file may be corrupted")

        if verify_crc32:
            crc = zlib.crc32(raw) & 0xFFFFFFFF
            if crc != e["crc32"]:
                raise ValueError("CRC32 mismatch: pack file may be corrupted")

        return raw


@contextlib.contextmanager
def open_resource_text(
    pack_path: str,
    name: str,
    encoding: str = "utf-8",
    errors: str = "strict",
):

    f = open(pack_path, "rb")
    try:
        entries, data_start = _read_index(f)
        if name not in entries:
            raise KeyError(f"resource not found: {name}")

        e = entries[name]
        f.seek(data_start + e["offset"])

        limited = LimitedReader(f, e["comp_len"])
        reader = _decompress_reader(limited, e["algo"])
        text = io.TextIOWrapper(reader, encoding=encoding, errors=errors, newline="")

        try:
            yield text
        finally:
            text.close()
    finally:
        f.close()

