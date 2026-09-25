"""
demo_visualizer.py — Interactive Visual Demonstration of the EdgeAI Satellite
Designed for live demos, student workshops, and science outreach.

Displays a live side-by-side satellite dashboard:
  1. Optical Camera View (Normal Human Vision)
  2. Thermal Heat Camera (Infrared Heat Vision)
  3. Fused Multi-Spectral Overlay (Combined Super-Vision)
  4. Live Satellite Onboard AI Decision Badge & Telemetry

Usage:
    python demo_visualizer.py [--sample 0] [--save]
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import digital_model_pipeline

def run_visual_demo(sample_idx: int = 0, save_fig: bool = False):
    manifest_path = os.path.join("data", "labeled", "manifest.csv")
    if not os.path.exists(manifest_path):
        print("Dataset not found. Generating proxy dataset first...")
        import package_dataset
        package_dataset.package_dataset()

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if sample_idx >= len(rows):
        sample_idx = 0

    row = rows[sample_idx]
    sid = row.get("sample_id", f"sample_{sample_idx:04d}")

    # 1. Load data
    rgb, thm, true_label = digital_model_pipeline.stage_load_sample(row)

    # 2. Run Edge AI Inference
    engine = digital_model_pipeline.EdgeInferenceEngine()
    pred_class, confidence, latency_ms = engine.predict(rgb, thm)

    # 3. Create visual images
    rgb_img = rgb[0]  # (128, 128, 3)
    thm_img = thm[0, :, :, 0]  # (32, 24)

    # Resize thermal for fusion visualization
    import cv2
    thm_resized = cv2.resize(thm_img, (128, 128), interpolation=cv2.INTER_LINEAR)
    fused_img = 0.5 * rgb_img + 0.5 * cv2.merge([thm_resized] * 3)
    fused_img = np.clip(fused_img, 0.0, 1.0)

    # 4. Setup Dark-Mode Satellite Dashboard
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor("#0b0f19")

    # Title Banner
    pred_text = "[CLOUD DETECTED]" if pred_class == 1 else "[CLEAR GROUND / VEGETATION]"
    badge_color = "#38bdf8" if pred_class == 1 else "#4ade80"

    fig.suptitle(
        f"EdgeAI Satellite Live Pass: {sid}  |  Altitude: 500 km  |  Speed: 7.6 km/s",
        fontsize=18, fontweight="bold", color="#f8fafc", y=0.96
    )

    # Subplot 1: Optical RGB
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb_img)
    ax1.set_title("1. Optical Camera\n(What Human Eyes See)", fontsize=13, color="#94a3b8", pad=10)
    ax1.axis("off")
    ax1.patch.set_edgecolor("#334155")
    ax1.patch.set_linewidth(2)

    # Subplot 2: Thermal Infrared
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(thm_img, cmap="inferno", aspect="auto")
    ax2.set_title("2. Thermal Heat Camera\n(Heat Vision: Bright = Warm, Dark = Cold)", fontsize=13, color="#94a3b8", pad=10)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8, label="Relative Temperature (DN)")

    # Subplot 3: Fused Super-Vision
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(fused_img)
    ax3.set_title("3. Fused Multi-Spectral\n(Combined Space Vision)", fontsize=13, color="#94a3b8", pad=10)
    ax3.axis("off")

    # Bottom Dashboard Info Banner
    info_box = (
        f">> ONBOARD AI DECISION:  {pred_text}\n"
        f"------------------------------------------------------------\n"
        f"* AI Confidence       : {confidence*100:.1f}%\n"
        f"* Processing Time     : {latency_ms:.2f} milliseconds (Faster than the blink of an eye!)\n"
        f"* Neural Network Size : 42.6 Kilobytes (Fits inside a microchip)\n"
        f"* Ground Coverage     : 62.9 km Swath over Target Region"
    )

    plt.figtext(
        0.5, 0.05, info_box,
        fontsize=12, family="monospace", color="#f1f5f9",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor=badge_color, linewidth=2.5)
    )

    plt.tight_layout(rect=[0.02, 0.22, 0.98, 0.92])

    out_png = f"demo_satellite_{sid}.png"
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"\n[DEMO READY] Saved demo screen to {out_png}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdgeAI Satellite Visual Demonstration")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to display (0-103)")
    args = parser.parse_args()
    run_visual_demo(sample_idx=args.sample)
