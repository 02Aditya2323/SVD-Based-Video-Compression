"""
Video Compression using Singular Value Decomposition (SVD)
Linear Algebra & Optimization Mini Project

Core idea:
  Each video frame = matrix A (H x W per color channel)
  SVD: A = U * diag(S) * V^T   (exact decomposition)
  Truncated SVD: A_k = U[:,:k] * diag(S[:k]) * V[:k,:]   (low-rank approximation)

  Keeping only k singular values/vectors compresses the frame.
  The singular values decay rapidly, so small k captures most of the "energy".

Usage:
  python svd.py --input your_video.mp4              # runs with default k=5
  python svd.py --input your_video.mp4 --rank 10   # run with a custom rank
  python svd.py                                      # auto-generates a test video
"""

import numpy as np
import cv2
import os
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse.linalg import svds


# ==============================================================
#  CORE SVD MATH
# ==============================================================

def svd_compress_channel(channel, k):
    """
    Low-rank approximation of a 2D matrix via TRUNCATED SVD.

    Instead of computing all singular values (np.linalg.svd),
    scipy.sparse.linalg.svds computes ONLY the top-k values.
    This is the main speedup — no wasted computation.

    Uses float32 instead of float64: visually identical, faster.

    Full SVD:     A = U * diag(S) * V^T
    Truncated:    A_k = U[:,:k] * diag(S[:k]) * V[:k,:]
    Storage:      k*(H + 1 + W)  instead of  H*W  values

    Args:
        channel : 2D float32 array (H x W)
        k       : number of singular values to keep
    Returns:
        Reconstructed 2D float32 array clipped to [0, 255]
    """
    k = min(k, min(channel.shape) - 1)  # svds requires k < min(H, W)
    try:
        # svds returns in ASCENDING order — reverse to get largest first
        U, S, Vt = svds(channel, k=k)
        U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1, :]
    except Exception:
        # Fallback: full SVD if svds fails (e.g., very small blocks)
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        S, U, Vt = S[:k], U[:, :k], Vt[:k, :]

    reconstructed = (U * S) @ Vt
    return np.clip(reconstructed, 0, 255)


def svd_compress_frame(frame, k):
    """
    Compress a full BGR color frame.
    SVD is applied independently to each color channel (B, G, R).
    All 3 channels are processed IN PARALLEL using threads.

    Args:
        frame : H x W x 3 uint8 numpy array (BGR)
        k     : SVD rank
    Returns:
        Compressed uint8 frame
    """
    # float32 is faster than float64 and sufficient for visual quality
    channels = [frame[:, :, c].astype(np.float32) for c in range(3)]

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda ch: svd_compress_channel(ch, k), channels))

    out = np.stack(results, axis=2).astype(np.uint8)
    return out


def compute_psnr(original, compressed):
    """
    Peak Signal-to-Noise Ratio (dB).
    > 40 dB   excellent (near-lossless visually)
    30-40 dB  good
    20-30 dB  acceptable
    < 20 dB   noticeable degradation
    """
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    if mse == 0:
        return 60.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


def compression_ratio(H, W, k):
    """
    Theoretical per-channel compression ratio.
    Original storage: H*W
    SVD storage:      k*(H + 1 + W)
    """
    return (H * W) / (k * (H + W + 1))


# ==============================================================
#  STREAMING COMPRESSION  (no full video loaded into RAM)
# ==============================================================

def compress_video_streaming(input_path, output_path, k, n_psnr_samples=120):
    """
    Stream through the ENTIRE video frame by frame:
      - Read one frame -> compress with truncated SVD rank-k -> write to output
      - Never loads all frames into RAM
      - Samples ~n_psnr_samples frames to compute average PSNR
      - Shows live progress bar with ETA

    Returns: (mean_psnr, ratio, elapsed_sec, fps, W, H, total_frames)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_idx = set(
        np.linspace(0, total - 1, min(n_psnr_samples, total), dtype=int).tolist()
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    psnr_list = []
    t0  = time.time()
    idx = 0
    BAR = 36

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        comp = svd_compress_frame(frame, k)
        writer.write(comp)

        if idx in sample_idx:
            psnr_list.append(compute_psnr(frame, comp))

        # Live progress bar
        pct     = (idx + 1) / max(total, 1)
        filled  = int(BAR * pct)
        bar_str = '#' * filled + '-' * (BAR - filled)
        elapsed = time.time() - t0
        eta     = (elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0
        avg_p   = np.mean(psnr_list) if psnr_list else 0.0
        print(f"\r    [{bar_str}] {idx+1}/{total}  "
              f"PSNR~{avg_p:.1f}dB  ETA:{int(eta)}s   ",
              end='', flush=True)
        idx += 1

    cap.release()
    writer.release()
    print()  # newline after progress bar

    elapsed   = time.time() - t0
    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    ratio     = compression_ratio(H, W, k)
    return mean_psnr, ratio, elapsed, fps, W, H, total


# ==============================================================
#  TEST VIDEO GENERATOR
# ==============================================================

def generate_test_video(path, n_frames=90, height=240, width=320):
    """Generate a synthetic moving-shapes video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 15, (width, height))
    print(f"  Generating test video ({n_frames} frames at {width}x{height})...")
    for i in range(n_frames):
        t = i / n_frames
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            r = int(255 * row / height)
            b = int(255 * (1 - row / height))
            g = int(128 + 127 * np.sin(2 * np.pi * (t + row / height)))
            frame[row, :] = [b, g, r]
        cx = int(width * (0.15 + 0.7 * t))
        cv2.circle(frame, (cx, height // 2), 28, (255, 255, 255), -1)
        rx = int(width * (0.85 - 0.7 * t))
        cv2.rectangle(frame,
                      (rx - 22, height // 2 - 22),
                      (rx + 22, height // 2 + 22),
                      (0, 200, 100), -1)
        out.write(frame)
    out.release()
    print(f"  Saved: {path}")


# ==============================================================
#  MAIN PIPELINE
# ==============================================================

def run_compression(input_path, output_path, rank, plot_path):
    """
    Compress the video at a single rank k.
    Saves the compressed video and generates a results plot.
    """
    print("\n" + "=" * 62)
    print("  SVD VIDEO COMPRESSION  |  Linear Algebra & Optimization")
    print("=" * 62)

    # Get video metadata
    cap   = cv2.VideoCapture(input_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    dur = total / fps if fps > 0 else 0
    print(f"\n  Input  : {input_path}")
    print(f"  Video  : {W}x{H}  |  {fps:.0f} fps  |  "
          f"{total} frames  |  {dur:.1f}s  ({dur/60:.1f} min)")
    print(f"  Rank k : {rank}  (theoretical ratio: {compression_ratio(H, W, rank):.1f}x)")
    print()

    temp_out = output_path.replace('.mp4', f'_k{rank}.mp4')

    mean_psnr, ratio, elapsed, *_ = compress_video_streaming(
        input_path, temp_out, rank, n_psnr_samples=150
    )

    quality = "excellent" if mean_psnr >= 40 else ("good" if mean_psnr >= 30 else "degraded")
    print(f"    Done in {elapsed:.0f}s  |  "
          f"Avg PSNR: {mean_psnr:.2f} dB [{quality}]  |  "
          f"Ratio: {ratio:.1f}x\n")

    # Move temp file to final output path
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_out, output_path)

    orig_mb = os.path.getsize(input_path)  / (1024 * 1024)
    comp_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Raw output (rank={rank}): {output_path}  ({comp_mb:.1f} MB)")

    # Re-compress with ffmpeg + H.264 for actual small file size
    final_output = output_path.replace('.mp4', '_final.mp4')
    ffmpeg_ok = False
    import shutil, subprocess
    if shutil.which('ffmpeg'):
        print(f"  Re-compres with ffmpeg (H.264)...")
        orig_kbps = int((os.path.getsize(input_path) * 8) / (1024 * dur) * 0.7)
        orig_kbps = max(orig_kbps, 100)
        print(f"  Target bitrate: {orig_kbps} kbps  (70% of original)")
        cmd = [
            'ffmpeg', '-y',
            '-i', output_path,
            '-vcodec', 'libx264',
            '-b:v', f'{orig_kbps}k',
            '-maxrate', f'{orig_kbps*2}k',
            '-bufsize', f'{orig_kbps*4}k',
            '-preset', 'fast',
            '-movflags', '+faststart',
            final_output
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(final_output):
            final_mb = os.path.getsize(final_output) / (1024 * 1024)
            os.remove(output_path)
            print(f"  Final output : {final_output}")
            print(f"  File size    ->  Original: {orig_mb:.1f} MB  |  SVD+H264: {final_mb:.1f} MB  |  Reduction: {orig_mb/final_mb:.1f}x")
            ffmpeg_ok = True
        else:
            print(f"  ffmpeg failed — keeping raw output.")
            print(result.stderr[-500:])
    if not ffmpeg_ok:
        print(f"  ffmpeg not found — keeping raw OpenCV output ({comp_mb:.1f} MB).")
        print(f"  Tip: install ffmpeg and re-run, or manually run:")
        print(f"    ffmpeg -i {output_path} -vcodec libx264 -crf 23 {final_output}")

    print(f"  Original file: {orig_mb:.1f} MB")

    # Plot
    print(f"\n  Generating results plot...")
    make_plot(mean_psnr, ratio, rank, H, W, plot_path)

    # Summary
    print("\n" + "=" * 62)
    print(f"  Rank k : {rank}")
    print(f"  Avg PSNR : {mean_psnr:.2f} dB  [{quality}]")
    print(f"  Compression Ratio : {ratio:.1f}x")
    print("=" * 62)
    print("\n  All done! Check output files.")


# ==============================================================
#  PLOT
# ==============================================================

def make_plot(mean_psnr, ratio, rank, H, W, save_path):
    """Single-rank summary plot: quality band + compression info."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#0f0f1a')

    tc = '#e0e0ff'
    bg = '#1a1a2e'
    c1 = '#00d4ff'
    c2 = '#ff6b9d'

    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=tc)
        for sp in ax.spines.values():
            sp.set_color('#333355')

    # Panel 1: PSNR quality gauge
    ax = axes[0]
    psnr_capped = min(mean_psnr, 60.0)
    ax.barh(['PSNR'], [psnr_capped], color=c1, height=0.4)
    ax.axvline(40, color='lime',   linestyle='--', lw=1.5, label='>40 dB excellent')
    ax.axvline(30, color='yellow', linestyle='--', lw=1.5, label='>30 dB good')
    ax.set_xlim(0, 65)
    ax.set_xlabel('PSNR (dB)', color=tc)
    ax.set_title(f'Quality at Rank k={rank}', color=tc, fontweight='bold')
    ax.legend(fontsize=8, facecolor=bg, labelcolor=tc)
    ax.annotate(f'{psnr_capped:.2f} dB', (psnr_capped, 0),
                xytext=(5, 0), textcoords='offset points',
                va='center', color=tc, fontsize=11, fontweight='bold')

    # Panel 2: Storage breakdown
    ax = axes[1]
    original_vals = H * W
    svd_vals      = rank * (H + W + 1)
    bars = ax.barh(['Original', f'SVD k={rank}'],
                   [original_vals, svd_vals],
                   color=[c2, c1], height=0.4)
    ax.set_xlabel('Values per channel (pixels)', color=tc)
    ax.set_title(f'Storage Comparison  ({ratio:.1f}x reduction)', color=tc, fontweight='bold')
    for bar, val in zip(bars, [original_vals, svd_vals]):
        ax.annotate(f'{val:,}', (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points',
                    va='center', color=tc, fontsize=9)

    fig.suptitle(f'SVD Video Compression  |  k={rank}  |  Results',
                 color=tc, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved: {save_path}")


# ==============================================================
#  ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVD Video Compression')
    parser.add_argument('--input',  type=str, default=None,
                        help='Input video (.mp4). Omit to auto-generate a test video.')
    parser.add_argument('--output', type=str, default='compressed_output.mp4',
                        help='Output compressed video path')
    parser.add_argument('--rank',   type=int, default=5,
                        help='SVD rank k to use for compression (default: 5). Higher = better quality, slower.')
    parser.add_argument('--plot',   type=str, default='svd_results.png',
                        help='Path to save the results plot')
    args = parser.parse_args()

    input_path = args.input
    if not input_path or not os.path.exists(input_path):
        input_path = 'test_video.mp4'
        print("[!] No input video found — generating a synthetic test video...")
        generate_test_video(input_path)

    run_compression(
        input_path  = input_path,
        output_path = args.output,
        rank        = args.rank,
        plot_path   = args.plot,
    )