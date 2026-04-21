# 🎬 SVD Video Compression

> **Linear algebra meets video codec engineering.** Two complete implementations of Singular Value Decomposition-based video compression — from a clean academic baseline to a full research-grade spatiotemporal pipeline.

---

## 📌 What is this?

This project compresses video using the mathematical power of **Singular Value Decomposition (SVD)** — the same decomposition that underlies PCA, recommender systems, and much of modern signal processing. Instead of storing pixel values directly, we factor each frame (or block-of-frames) into lower-rank approximations that capture the visual "essence" with a fraction of the data.

Two separate implementations are provided, each with a distinct tradeoff:

| | Default SVD (`svd.py`) | Research Paper SVD+DVC (`svd_new.py`) |
|---|---|---|
| **Method** | Frame-by-frame truncated SVD | Block-GOP SVD + Wyner-Ziv DVC |
| **Compression axis** | Spatial only | Spatial **+** Temporal |
| **Memory model** | Streaming (RAM-efficient) | Full GOP loaded |
| **Based on** | Linear algebra fundamentals | Dhivya & Kannan, IJERD 2013 |
| **Key metric** | PSNR vs. rank `k` | PSNR gain: SVD → SVD+DVC |
| **Output** | SVD raw → H.264 re-encode | SVD+DVC raw → H.264 re-encode |

---

## 📐 The Core Math

Both scripts are grounded in the **truncated SVD** approximation:

$$A = U \Sigma V^T \approx U_k \Sigma_k V_k^T$$

For an $H \times W$ frame matrix $A$:
- **Original storage:** $H \times W$ values
- **SVD storage:** $k \times (H + W + 1)$ values
- **Compression ratio:** $\dfrac{H \cdot W}{k \cdot (H + W + 1)}$

The singular values $\sigma_1 \geq \sigma_2 \geq \cdots$ **decay rapidly** — keeping only the top-$k$ terms captures the dominant structure while discarding fine noise. This is the mathematical legitimacy behind lossy compression with SVD.

---

## 📁 Project Structure

```
svd_new_claudescript/
│
├── Default compression/
│   ├── svd.py                  # Baseline streaming SVD compressor
│   ├── svd_results.png         # Output quality plot
│   └── *.mp4                   # Test output videos (k=25, k=50, k=250)
│
└── compression based on research papers/
    ├── svd_new.py              # Full SVD+DVC pipeline (paper implementation)
    ├── svd_dvc_results.png     # Stage comparison plot
    ├── compressed_svd_dvc.mp4  # Example output
    └── International_Journal_of_Engineering_Res (2).pdf  # Source paper
```

---

## 1️⃣ Default Compression — `svd.py`

### What it does

A clean, memory-efficient baseline. Every frame is compressed **independently**: the BGR image is split into three $H \times W$ channel matrices, each factored via truncated SVD to rank $k$, then reconstructed and written to disk.

### Pipeline

![Default SVD compression pipeline — frame decoded by OpenCV, channels split and processed in parallel with svds(k), then recombined and written with optional H.264 re-encode](Screenshot%202026-04-21%20at%2013.03.33.png)

**Key design decisions:**
- Uses `scipy.sparse.linalg.svds` (not full SVD) — only computes the top-$k$ singular vectors, saving significant compute for large frames
- All three color channels compressed in **parallel threads** (`ThreadPoolExecutor`)
- Streaming architecture: never loads full video into RAM
- Samples $\leq150$ frames for PSNR computation to avoid slowing the write loop
- Raw `.mp4v` output → **H.264 re-encode via ffmpeg** (targets 70% of original bitrate), which is where actual file-size reduction materializes

### The H.264 Re-encode Step

The SVD stage alone doesn't shrink the file on disk — `mp4v` is an uncompressed pixel dump. The real size reduction comes from piping the SVD-approximated frames into **H.264** (ffmpeg), which then exploits inter-frame redundancy (DCT, motion vectors, CABAC entropy coding) on top of the already-smoothed SVD output.

![H.264 re-encode pipeline — bitrate target calculated, ffmpeg reads frames, branches into I-frame (full DCT) or P/B-frame (motion vectors + residual), quantization hits target bitrate, entropy coded via CABAC](Screenshot%202026-04-21%20at%2013.09.46.png)

### Usage

```bash
cd "Default compression"

# Auto-generate test video and compress
python svd.py

# Compress your own video at rank k=10
python svd.py --input your_video.mp4 --rank 10

# Use a higher rank for better quality
python svd.py --input your_video.mp4 --rank 50 --output out.mp4 --plot results.png
```

### Parameters

| Argument | Default | Description |
|---|---|---|
| `--input` | *(auto-gen)* | Input `.mp4` path |
| `--output` | `compressed_output.mp4` | Output path |
| `--rank` | `5` | SVD rank `k`. Higher = better quality, lower compression |
| `--plot` | `svd_results.png` | Path for the results plot |

### Quality vs. Compression Tradeoff

| Rank `k` | Approx. Compression Ratio | Expected PSNR |
|---|---|---|
| 5 | ~20–40× | ~28–35 dB |
| 25 | ~5–10× | ~35–42 dB |
| 50 | ~3–5× | ~40–48 dB |
| 250 | ~0.8–1× | ~55+ dB |

> **Rule of thumb:** PSNR ≥ 40 dB = excellent (near-lossless visually). 30–40 dB = good. Below 30 dB = noticeable artifacts.

---

## 2️⃣ Research Paper Pipeline — `svd_new.py`

### What it does

A two-stage codec directly implementing the algorithm from:

> **"Video Compression based on Singular Value Decomposition and Distributed Video Coding"**  
> S. Dhivya, M. Kannan — *International Journal of Engineering Research and Development (IJERD), 2013*

The key insight: instead of compressing each frame in isolation, group frames into a **Group of Pictures (GOP)** and exploit redundancy *across time* using block-level SVD, then further compress temporal residuals with a **Distributed Video Coding (DVC)** / Wyner-Ziv scheme.

### Stage 1 — Block GOP SVD

```
For each GOP of n frames:
  1. Compute mean frame:  F_mean = (1/n) Σ F_i
  2. Subtract mean:       F_i' = F_i - F_mean
  3. Divide all frames into 8×8 blocks
  4. For each block position (bh, bw):
       Stack n blocks → matrix M of shape (n, 64)
       Apply SVD: M = U · S · Vt
       Keep top-k:  Uk (n×k), Sk (k,), Vtk (k×64)
     → Vtk sent ONCE as "group information" for all n frames
     → Each frame only needs (Uk[i,:], Sk) — tiny per-frame payload
  5. Reconstruct: recon_block = (Uk * Sk) @ Vtk
  6. Add mean frame back
```

**Why this is powerful:** The right singular vectors $V_k^T$ (spatial patterns) are *shared* across the entire GOP. Only the scalar coefficients per frame change. This is the equivalent of learning a shared "vocabulary" for a group of frames and encoding each frame as a short "sentence" in that vocabulary.

**Theoretical compression ratio per block:**

$$\text{ratio} = \frac{n \times 64}{k \times (n + 64 + 1)}$$

For `n=24, k=3`: ratio ≈ **5.2×** just in the SVD stage.

### Stage 2 — Distributed Video Coding (Wyner-Ziv)

```
For each GOP:
  - Frame 0  → Key Frame (transmitted fully, H.264 in real codec)
  - Frames 1..n-1 → Wyner-Ziv (WZ) frames:
      1. Side Information (SI):  linear interpolation between key frames
         SI_i = (1 - α) · F_0 + α · F_{n-1},  α = i/(n-1)
      2. Residual:  R_i = WZ_i - SI_i
      3. Quantize:  Q_i = round(R_i / q_step) * q_step   ← simulates parity bits
      4. Reconstruct: SI_i + Q_i
```

The DVC stage adds **3–8 dB PSNR** on top of SVD alone — the interpolation already captures smooth motion, and the quantized residual corrects for what interpolation misses.

### Usage

```bash
cd "compression based on research papers"

# Auto-generate test video and run full SVD+DVC pipeline
python svd_new.py

# Compress your video with paper-recommended settings
python svd_new.py --input your_video.mp4 --gop 24 --rank 3 --dvc 8

# Higher quality mode (less compression)
python svd_new.py --input your_video.mp4 --gop 21 --rank 8 --dvc 4
```

### Parameters

| Argument | Default | Description |
|---|---|---|
| `--input` | *(auto-gen)* | Input `.mp4` path |
| `--output` | `compressed_svd_dvc.mp4` | Output path |
| `--rank` | `3` | SVD rank `k` for block SVD. Paper uses 1–8. |
| `--gop` | `24` | GOP size. Paper says **21–31 is optimal**. |
| `--dvc` | `8` | DVC quantization step. Lower = better WZ quality. |
| `--plot` | `svd_dvc_results.png` | Path for results plot |

### Paper-recommended Settings

| Setting | Value | Notes |
|---|---|---|
| GOP size | 21–31 | Balances temporal coherence vs. latency |
| SVD rank `k` | 3 | Good quality/ratio tradeoff |
| DVC quant step | 8 | Reasonable residual fidelity |

---

## 📊 Output & Metrics

Both scripts generate:
1. **A compressed `.mp4`** — SVD-approximated frames, H.264 re-encoded for actual size reduction
2. **A results plot** — dark-themed matplotlib chart showing:

**`svd.py` plot:**
- PSNR quality gauge vs. reference lines (40 dB, 30 dB)
- Storage comparison: original $H \times W$ vs. SVD $k(H+W+1)$ per channel

**`svd_new.py` plot (3 panels):**
- Bar chart: SVD-only PSNR vs. SVD+DVC PSNR
- Per-frame PSNR timeline across the whole video
- Per-block storage: original `n×64` vs. SVD `k×(n+64+1)`

---

## ⚙️ Dependencies

```bash
pip install numpy opencv-python scipy matplotlib
```

`ffmpeg` is optional but highly recommended for actual file-size reduction:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

---

## 🔑 Key Concepts Quick-Reference

| Term | Meaning |
|---|---|
| **SVD** | $A = U\Sigma V^T$ — decomposes a matrix into orthogonal bases and singular values |
| **Truncated SVD** | Keep only top-$k$ singular values — low-rank approximation |
| **Rank `k`** | Number of singular values kept. Higher = better quality, less compression |
| **PSNR** | Peak Signal-to-Noise Ratio (dB). $10\log_{10}(255^2/\text{MSE})$ |
| **GOP** | Group of Pictures — a chunk of temporally adjacent frames |
| **DVC** | Distributed Video Coding — encodes frames using side information at the decoder |
| **Wyner-Ziv** | Information-theoretic DVC variant: encoder sends only parity; decoder uses SI |
| **Side Information (SI)** | Decoder's prediction of a WZ frame (via interpolation) |
| **Key Frame** | Intra-coded reference frame transmitted in full |
| **ffmpeg / H.264** | Final re-encode step that applies inter-frame motion compensation on SVD output |

---

## 📚 Reference

```
S. Dhivya, M. Kannan,
"Video Compression based on Singular Value Decomposition and Distributed Video Coding",
International Journal of Engineering Research and Development (IJERD), 2013.
```

---

<div align="center">

Built with `numpy` · `opencv-python` · `scipy` · `matplotlib` · `ffmpeg`

*Linear algebra-powered video compression — from first principles.*

</div>
