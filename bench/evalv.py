"""evalv.py NAME BINA OPTSA -- BINB OPTSB : paired film-referenced comparison on all spans (edge/flat errors)."""
import sys, json, numpy as np, bench as B, prtool as P
def evaluate(a, b, films=("diner", "shrek")):
    out = {}
    for film in films:
        rows = []
        for base in B.SPAN_FILMS[film]:
            sp = B.SPANS[base]; n = sp["n"]; l = P.luma(base, 1, n); valid = B.picture_mask(base) > 0
            ca = P.chroma(a[0], base, 1, n, a[1]); cb = P.chroma(b[0], base, 1, n, b[1])
            for i in B.good_frames(base):
                r = B.ref_cached(base, i)
                ea, _ = B.frame_errors(P.yuv(l, ca, i), r, valid); eb, _ = B.frame_errors(P.yuv(l, cb, i), r, valid)
                rows.append([ea["edge"], ea["flat"], eb["edge"], eb["flat"]])
        R = np.array(rows); d = R[:, 2:] / R[:, :2] - 1
        rng = np.random.default_rng(1); bs = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(4000)])
        out[film] = dict(n=len(R), a=R[:, :2].mean(0).tolist(), b=R[:, 2:].mean(0).tolist(), change=d.mean(0).tolist(),
                         ci=np.percentile(bs, [2.5, 97.5], axis=0).tolist(), better=(d < 0).mean(0).tolist())
    return out
if __name__ == "__main__":
    pass
