# SECAM decoder PRs: figures and bench

Figures referenced by the SECAM pull requests to harrypm/tbc-tools, and the bench that measured them.

Nothing here is meant to be merged.

## Material

The material comes from two films released on SECAM VHS: *Le Dîner de Cons* (1998) and *Shrek* (2001).

The tapes were played on a Panasonic NV-HD647F and the RF was captured at 40 MSPS (cxadc). The RF was decoded with tape-decode-rs, profile `SECAM_VHS`. The test spans are:

- Le Dîner de Cons: five spans of 22 s each.
- Shrek: one span of 30 s.

The RF captures are not published. The stills are small crops used only to compare decoders.

## Bench (`bench/`)

### Reference

The reference is the Blu-ray of the same film, aligned to every 20th VHS frame:

- **Time:** from the known PAL speed-up (25/24, or 25/23.976), refined on three textured frames.
- **Space:** a fixed scale per tape, with a per-frame translation found by FFT correlation of the luma gradients. Frames scoring below 0.35 are dropped, which leaves 106 frames for the Diner and 25 for Shrek.
- **Colour:** the film is converted to BT.601 Y'CbCr, the units of the VHS decode.

### Errors

Every error is the colour-difference error, |ΔCb, ΔCr|, measured after a low-frequency 2x2 + offset colour fit (σ = 6 px). The fit removes what a global colour setting could fix, and keeps placement errors. The error is reported in three regions:

| Region | Where it is measured |
|---|---|
| `edge` | RMS within ±4 px of the film's strongest 8 % colour gradients |
| `halo` | RMS in the 4–20 px ring around them, where colour bleed lands |
| `flat` | RMS in the film's calm half, away from the edges |

Intervals are 95 % paired-bootstrap intervals over frames (Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993).

### Step response

The step response is measured without any reference. At a sharp luma edge with calm luma on both sides (std ≤ 0.025 over 60 to 80 samples) and a colour change ≥ 0.04, the median normalised colour profile is the colour chain's step response.

### Running it

Set these environment variables:

- `PR_ROOT`
- `DINER_FILM` and `SHREK_FILM`
- `CHROMA_MAIN` (a build of main) and `CHROMA_<NAME>` (one build per PR branch; `LABO` is all the branches merged)
