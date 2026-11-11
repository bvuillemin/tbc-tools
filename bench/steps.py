"""Colour step response at clean luma edges (no reference needed), after lab/bleed.py:step_profiles.
At a sharp luma edge with calm flats on both sides and a real colour change, the true colour steps exactly
at the luma edge, so the decoded colour profile across it is the chain's step response."""
import numpy as np
from scipy import ndimage

def step_profiles(y, c, pre=60, post=60, gthr=0.05, athr=0.04, calm=0.025, rows=(40, 536)):
    out = []
    ys = ndimage.uniform_filter1d(y, 3, axis=1)
    W = y.shape[1]
    for r in range(rows[0], rows[1]):
        g = np.abs(ys[r, 2:] - ys[r, :-2]) / 2
        cand = np.nonzero(g > gthr)[0] + 1
        cand = [x for x in cand if g[x - 1] >= g[x - 2] and g[x - 1] >= g[min(x, len(g) - 1)]]
        for x in cand:
            if x < 100 or x > W - 100: continue
            if y[r, x - pre:x - 4].std() > calm or y[r, x + 5:x + post].std() > calm: continue
            a0 = c[r, x - pre:x - pre + 12].mean(); a1 = c[r, x + post - 12:x + post].mean()
            amp = a1 - a0
            if abs(amp) < athr: continue
            out.append((c[r, x - pre:x + post] - a0) / amp)
    return np.array(out).reshape(-1, pre + post)
