"""Film-referenced bench for the SECAM PRs.

For a decoded VHS span (TBC base name + frame range) and the matching Blu-ray of the same film:
  1. time: the VHS mean-luma sequence is cross-correlated with the film's (film played at 25/24 for the
     Diner, 25/23.976 for Shrek) to find the film frame shown by each VHS frame;
  2. space: the film frame is warped onto the VHS sampling grid (x_vhs = ax*x_f + bx, y_vhs = ay*y_f + by),
     scale found once per span on a textured frame, translation per frame (FFT correlation of luma gradients);
  3. colour: the film is converted to BT.601 Y'CbCr, the units of the VHS decode.

Alignment results are cached in cache/align_<base>.json.
"""
import os, json, subprocess, sys
import numpy as np
from scipy import ndimage
import prtool as P

W, H = P.W, P.H
RW, RH = 960, 540
FILMS = {
    "diner": dict(path=os.environ.get("DINER_FILM", "diner-bluray.mkv"),
                  fps=24.0, rows=(128, 448), size=(960, 410)),
    "shrek": dict(path=os.environ.get("SHREK_FILM", "shrek-bluray.mkv"),
                  fps=24000 / 1001, rows=None, size=(960, 540)),
}
GEOM = {"diner": (0.9728, 0.785), "shrek": (1.2177, 0.99)}
SPANS = {
    "shrek": dict(film="shrek", n=880, guess=88708 / (24000 / 1001)),
    **{f"diner{t}": dict(film="diner", n=545, guess=(t - 263.60) / 0.96) for t in (795, 1500, 2560, 3300, 4000)},
}
# same spans decoded by the upstream tape-decode-rs (same RF, same frames)
SPANS.update({k + ".up": v for k, v in list(SPANS.items()) if k in ("shrek", "diner795", "diner2560")})


def film_path(name):
    p = FILMS[name]["path"]
    if os.path.isdir(p):
        p = os.path.join(p, [f for f in os.listdir(p) if f.endswith(".mkv")][0])
    return p


def film_frames(name, t0, n, w=None, h=None, gray=False):
    """n consecutive film frames from time t0 (s), as (Y, Cb, Cr) float32 BT.601 or gray only."""
    if w is None:
        w, h = FILMS[name]["size"]
    fmt = "gray" if gray else "rgb48le"
    vf = f"scale={w}:{h}:flags=area:in_color_matrix=bt709,format={fmt}"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t0 - 1):.4f}", "-i", film_path(name),
                          "-ss", f"{min(1, t0):.4f}", "-frames:v", str(n), "-map", "0:v:0", "-vf", vf,
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    if gray:
        return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32) / 255
    rgb = np.frombuffer(raw, np.uint16).reshape(-1, h, w, 3).astype(np.float32) / 65535
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return [(y[i], (b[i] - y[i]) / 1.772, (r[i] - y[i]) / 1.402) for i in range(len(y))]


def vhs_luma_seq(base, n):
    a = P.frames(P.luma(base, 1, n))
    return np.array([a[i, 0, 140:440:4, 60:880:4].astype(np.float32).mean() for i in range(len(a))])


def time_align(base):
    """Film time of VHS frame 0 and film frames per VHS frame, from the mean-luma sequences."""
    sp = SPANS[base]; f = FILMS[sp["film"]]
    v = vhs_luma_seq(base, sp["n"])
    rate = f["fps"] / 25.0                     # film frames advanced per VHS frame (VHS plays the film at 25)
    if sp["guess"] is None:
        t0, span = 3699.0, 60.0                # Shrek 30 s capture: film ~61:39 (F0 ~ 88708 / 23.976)
    else:
        t0, span = sp["guess"], 12.0
    fl = film_frames(sp["film"], t0 - span, int((2 * span + len(v) / 25 * 1.1) * f["fps"]), 64, 36, gray=True)
    fs = fl[:, 4:-4, 4:-4].mean((1, 2))
    best = None
    for k0 in range(0, len(fs) - int(len(v) * rate) - 1):
        idx = k0 + np.arange(len(v)) * rate
        s = np.interp(idx, np.arange(len(fs)), fs)
        c = np.corrcoef(np.diff(s), np.diff(v))[0, 1]
        if best is None or c > best[0]:
            best = (float(c), k0)
    c, k0 = best
    return dict(corr=c, film_t0=t0 - span + k0 / f["fps"], rate=rate)


def warp(img, ax, ay, bx, by, out_rows=None):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return ndimage.map_coordinates(img, [(yy - by) / ay, (xx - bx) / ax], order=1, mode="nearest")


def grad(p):
    return np.hypot(*np.gradient(ndimage.gaussian_filter(p, 1.2)))


def align_space(yv, yf, grid, mask):
    gv = grad(yv) * mask; gv -= gv[mask > 0].mean(); gv *= mask
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    Fv = np.fft.rfft2(gv * win)
    best = None
    for ax, ay in grid:
        g = grad(warp(yf, ax, ay, 0, 0)); g -= g.mean()
        cc = np.fft.irfft2(np.fft.rfft2(g * win).conj() * Fv, s=(H, W))
        k = np.unravel_index(np.argmax(cc), cc.shape)
        v = cc[k] / (np.linalg.norm(g * win) * np.linalg.norm(gv * win) + 1e-9)
        dy = k[0] if k[0] < H // 2 else k[0] - H
        dx = k[1] if k[1] < W // 2 else k[1] - W
        if best is None or v > best[0]:
            best = (float(v), float(ax), float(ay), float(dx), float(dy))
    return best


def picture_mask(base):
    m = np.zeros((H, W), np.float32)
    rows = FILMS[SPANS[base]["film"]]["rows"] or (20, 560)
    m[rows[0] + 6:rows[1] - 6, 30:W - 40] = 1
    return m


def alignment(base, step=20):
    path = f"{P.CACHE}/align_{base}.json"
    if os.path.exists(path):
        return json.load(open(path))
    sp = SPANS[base]; fps = FILMS[sp["film"]]["fps"]; rate = fps / 25.0
    lpath = P.luma(base, 1, sp["n"])
    mask = picture_mask(base)
    ax, ay = GEOM[sp["film"]]
    vy = lambda i: (P.frames(lpath)[i, 0].astype(np.float32) / 256 - 16) / 219
    sel = list(range(10, sp["n"] - 10, step))
    # time: VHS frame i shows film time t0 + i*rate/fps; t0 refined over +-0.6 s on three textured frames
    tex = sorted(sel, key=lambda i: -grad(vy(i))[mask > 0].mean())[:3]
    best = None
    for d in range(-14, 15):
        t0 = sp["guess"] + d / fps
        sc = [align_space(vy(i), film_frames(sp["film"], t0 + i * rate / fps + 0.25 / fps, 1)[0][0],
                          [(ax, ay)], mask)[0] for i in tex]
        if best is None or np.mean(sc) > best[0]:
            best = (float(np.mean(sc)), t0)
    t0 = best[1]
    print(base, "t0", t0, "score", best[0], flush=True)
    res = dict(time=dict(film_t0=t0, rate=rate, score=best[0]), ax=ax, ay=ay, frames={})
    for i in sel:
        t = t0 + i * rate / fps
        r = align_space(vy(i), film_frames(sp["film"], t + 0.25 / fps, 1)[0][0], [(ax, ay)], mask)
        res["frames"][str(i)] = dict(score=r[0], bx=r[3], by=r[4], t=t)
    json.dump(res, open(path, "w"), indent=1)
    return res


def reference(base, i, al=None):
    """Film frame matching VHS frame i, warped onto the VHS grid: (Y, Cb, Cr)."""
    al = al or alignment(base)
    fr = al["frames"][str(i)]
    f = film_frames(SPANS[base]["film"], fr["t"] + 0.25 / FILMS[SPANS[base]["film"]]["fps"], 1)[0]
    return [warp(p, al["ax"], al["ay"], fr["bx"], fr["by"]) for p in f]


if __name__ == "__main__":
    for b in sys.argv[1:]:
        a = alignment(b)
        sc = [v["score"] for v in a["frames"].values()]
        print(b, "time score %.3f" % a["time"]["score"], "scale %.3f %.3f" % (a["ax"], a["ay"]),
              "scores median %.2f min %.2f" % (np.median(sc), min(sc)))


SPAN_FILMS = {"diner": [f"diner{t}" for t in (795, 1500, 2560, 3300, 4000)], "shrek": ["shrek"]}
MINSCORE = 0.35


def good_frames(base):
    al = alignment(base)
    return [int(k) for k, v in al["frames"].items() if v["score"] >= MINSCORE]


def lp(p, s=6):
    return ndimage.gaussian_filter(p, s)


def colour_fit(cb, cr, cbr, crr, valid, diagonal=False):
    """Low-frequency fit (VHS -> film): full 2x2 + offset, or per-component gain + offset."""
    sub = lambda p: lp(p)[valid][::16]
    X = np.stack([sub(cb), sub(cr), np.ones_like(sub(cb))], 1)
    if diagonal:
        gb = np.linalg.lstsq(X[:, [0, 2]], sub(cbr), rcond=None)[0]
        gr = np.linalg.lstsq(X[:, [1, 2]], sub(crr), rcond=None)[0]
        return np.array([[gb[0], 0], [0, gr[0]], [gb[1], gr[1]]])
    return np.linalg.lstsq(X, np.stack([sub(cbr), sub(crr)], 1), rcond=None)[0]


def apply_fit(Pm, cb, cr):
    return Pm[0, 0] * cb + Pm[1, 0] * cr + Pm[2, 0], Pm[0, 1] * cb + Pm[1, 1] * cr + Pm[2, 1]


def cgrad(cb, cr, s=1.0):
    g = lambda p: ndimage.gaussian_filter(p, s)
    return np.sqrt(sum(np.hypot(*np.gradient(g(p))) ** 2 for p in (cb, cr)))


def regions(ref, valid):
    """Film colour-edge band (+-4 px around the strongest 8 % colour gradients) and flat areas."""
    E = cgrad(ref[1], ref[2])
    thr = np.percentile(E[valid], 92)
    core = (E > thr) & valid
    band = ndimage.binary_dilation(core, iterations=4) & valid
    flat = (E < np.percentile(E[valid], 50)) & ~ndimage.binary_dilation(core, iterations=8) & valid
    return band, flat


def halo_ring(ref, valid):
    """Ring 4..20 px around the film's strong colour edges (where colour bleed lands)."""
    E = cgrad(ref[1], ref[2])
    core = (E > np.percentile(E[valid], 92)) & valid
    return ndimage.binary_dilation(core, iterations=20) & ~ndimage.binary_dilation(core, iterations=4) & valid


def frame_errors(vhs, ref, valid, Pm=None):
    """Colour error after removing what a global colour setting could fix (2x2 + offset, low freq)."""
    y, cb, cr = vhs
    if Pm is None:
        Pm = colour_fit(cb, cr, ref[1], ref[2], valid)
    fcb, fcr = apply_fit(Pm, cb, cr)
    D = np.hypot(fcb - ref[1], fcr - ref[2])
    band, flat = regions(ref, valid)
    ring = halo_ring(ref, valid)
    return dict(edge=float(np.sqrt((D[band] ** 2).mean())), flat=float(np.sqrt((D[flat] ** 2).mean())),
                halo=float(np.sqrt((D[ring] ** 2).mean()))), D


_REF = {}


def ref_cached(base, i):
    k = (base, i)
    if k not in _REF:
        _REF[k] = reference(base, i)
    return _REF[k]


def comparison(base, i, box, variants, zoom=2, errmax=0.12, show_err=True, film=True):
    """Crops (box = x0, y0, x1, y1 in VHS frame samples/lines) of the film and of each variant
    [(label, binary, opts)], one field, square pixels, Lanczos zoom; second row: residual colour error
    vs the film after the global low-frequency colour fit (the same map the edge/flat numbers use)."""
    n = SPANS[base]["n"]; l = P.luma(base, 1, n); valid = picture_mask(base) > 0
    ref = ref_cached(base, i)
    x0, y0, x1, y1 = box
    def crop(img):
        f = P.field(img)[y0:y1, x0:x1]
        f = P.square(f)
        from PIL import Image
        return np.asarray(Image.fromarray(f).resize((f.shape[1] * zoom, f.shape[0] * zoom), Image.LANCZOS))
    top = [P.label(crop(P.rgb(*ref)), "Blu-ray")] if film else []
    bot = [P.label(crop(np.full((H, W, 3), 255, np.uint8)), "error vs film:")] if film else []
    for var in variants:
        name, binn, opts = var[:3]
        v = P.yuv(l, P.chroma(binn, base, 1, n, opts), i)
        if len(var) > 3:
            v = (v[0], var[3](v[1]), var[3](v[2]))
        e, D = frame_errors(v, ref, valid)
        top.append(P.label(crop(P.rgb(*v)), name))
        bot.append(P.label(crop(P.heat(D, errmax)), "edge %.4f  flat %.4f" % (e["edge"], e["flat"])))
    return [top, bot] if show_err else [top]
