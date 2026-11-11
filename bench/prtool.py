"""Helpers for the PR comparisons: decode a TBC span with a given tbc-chroma-decoder binary and options,
load frames as normalised (Y 0..1, Cb/Cr +-0.5) float32 arrays, and make comparison images."""
import os, subprocess, hashlib, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 928, 576
ROOT = os.environ.get("PR_ROOT", ".")
TBC = f"{ROOT}/tbc"
CACHE = f"{ROOT}/cache"
BIN = {
    "upstream": os.environ.get("CHROMA_MAIN", "tbc-chroma-decoder.main"),
}
os.makedirs(CACHE, exist_ok=True)


def binary(name):
    return BIN.get(name, os.environ.get("CHROMA_" + name.upper(), f"tbc-chroma-decoder.{name}"))


def _run(args, out):
    if not os.path.exists(out):
        tmp = out + ".part"
        r = subprocess.run(args + [tmp], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(" ".join(args) + "\n" + r.stderr[-2000:])
        os.replace(tmp, out)
        with open(out + ".log", "w") as f:
            f.write(r.stderr)
    return out


def luma(base, start, length):
    out = f"{CACHE}/{base}_{start}_{length}_Y.yuv"
    return _run([BIN["upstream"], "-f", "mono", "--chroma-gain", "0", "-p", "yuv", "-t", "8",
                 "--input-json", f"{TBC}/{base}.tbc.json", "-s", str(start), "-l", str(length),
                 f"{TBC}/{base}.tbc"], out)


def chroma(binname, base, start, length, opts=()):
    opts = [o.replace("@LUMA", f"{TBC}/{base}.tbc") for o in opts]
    key = hashlib.md5((binname + " " + " ".join(opts)).encode()).hexdigest()[:10]
    out = f"{CACHE}/{base}_{start}_{length}_{binname}_{key}_C.yuv"
    return _run([binary(binname), "-f", "secam", "-p", "yuv", "-t", "12",
                 "--input-json", f"{TBC}/{base}.tbc.json", "-s", str(start), "-l", str(length)]
                + list(opts) + [f"{TBC}/{base}_chroma.tbc"], out)


def frames(path):
    a = np.memmap(path, np.uint16, "r")
    return a.reshape(-1, 1 if path.endswith("_Y.yuv") else 3, H, W)


def yuv(lpath, cpath, i):
    """Frame i -> (Y, Cb, Cr) float32, Y 0..1 (black..white), Cb/Cr +-0.5 full scale."""
    y = frames(lpath)[i, 0].astype(np.float32)
    c = frames(cpath)[i]
    y = (y / 256 - 16) / 219
    cb = (c[1].astype(np.float32) / 256 - 128) / 224
    cr = (c[2].astype(np.float32) / 256 - 128) / 224
    return y, cb, cr


def rgb(y, cb, cr):
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], -1) * 255 + 0.5, 0, 255).astype(np.uint8)


def field(img, parity=0):
    """One field (every other line), line-doubled back to frame height with Lanczos (no combing)."""
    f = img[parity::2]
    return np.asarray(Image.fromarray(f).resize((f.shape[1], f.shape[0] * 2), Image.LANCZOS))


def square(img):
    """Resample to square pixels (928 samples span 52 us of a 4:3 line -> 768 px)."""
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize((round(w * 259 / 311), h), Image.LANCZOS))


_FONT = None


def font(size=22):
    global _FONT
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def label(img, text, size=22):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    f = font(size)
    x0, y0, x1, y1 = d.textbbox((8, 6), text, font=f)
    d.rectangle((x0 - 5, y0 - 4, x1 + 5, y1 + 4), fill=(0, 0, 0))
    d.text((8, 6), text, font=f, fill=(255, 255, 255))
    return np.asarray(im)


def grid(rows, gap=6):
    """rows: list of lists of HxWx3 uint8 images (same size within a row)."""
    rimgs = []
    for r in rows:
        h = max(i.shape[0] for i in r)
        parts = []
        for k, i in enumerate(r):
            if k:
                parts.append(np.full((h, gap, 3), 255, np.uint8))
            parts.append(i)
        rimgs.append(np.concatenate(parts, 1))
    w = max(r.shape[1] for r in rimgs)
    out = []
    for k, r in enumerate(rimgs):
        if k:
            out.append(np.full((gap, w, 3), 255, np.uint8))
        if r.shape[1] < w:
            r = np.concatenate([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)], 1)
        out.append(r)
    return np.concatenate(out, 0)


def save(img, path, scale=1.0):
    im = Image.fromarray(img)
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, optimize=True)
    return path


def heat(a, vmax, cmap="magma"):
    """Scalar map -> RGB with a perceptual colormap (matplotlib)."""
    import matplotlib
    m = matplotlib.colormaps[cmap]
    return (m(np.clip(a / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
