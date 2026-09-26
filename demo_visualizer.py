"""
demo_visualizer.py — Interactive Multi-Scene Visual Demonstrator for EdgeAI Satellite
Supports:
  1. Scene 1 (Delhi Pass): Optical Cloud Filtering & Vegetation Mapping
  2. Scene 2 (Siberia Pass): Wildfire / Thermal Anomaly Front Detection
  3. Interactive Slideshow with Keyboard Controls (Pause, Step Forward/Back, Custom Delay)
  4. PNG Export

Keyboard Controls in Slideshow Mode:
  - Spacebar        : Pause / Resume slideshow
  - Right Arrow / D : Step forward to next sample
  - Left Arrow  / A : Step backward to previous sample
  - Q / Esc         : Exit cleanly

Usage:
    python demo_visualizer.py [--scene delhi|siberia] [--sample N] [--slideshow] [--delay 5.0] [--save]
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

from load_aggregated_data import (
    load_delhi_data,
    load_siberia_fire_data
)
from digital_model_pipeline import EdgeInferenceEngine

MODELS_DIR = "models"


class VisualizerController:
    def __init__(self, initial_idx: int = 0, total_samples: int = 100, is_slideshow: bool = False):
        self.idx = initial_idx
        self.total = total_samples
        self.paused = False
        self.is_slideshow = is_slideshow
        self.running = True
        self.manual_step = False

    def on_key(self, event):
        if event.key in ["q", "escape"]:
            self.running = False
            plt.close("all")
        elif event.key == " ":
            self.paused = not self.paused
            print(f"[{'PAUSED' if self.paused else 'RESUMED'}] Press Space to toggle, Right/Left arrows to step.")
        elif event.key in ["right", "d", "n"]:
            self.idx = (self.idx + 1) % self.total
            self.manual_step = True
        elif event.key in ["left", "a", "p"]:
            self.idx = (self.idx - 1 + self.total) % self.total
            self.manual_step = True

    def on_close(self, event):
        self.running = False


def run_visual_demo(
    scene: str = "siberia",
    sample_idx: int = 0,
    save_fig: bool = False,
    slideshow: bool = False,
    delay_s: float = 5.0
):
    # Load scene data
    if scene.lower() == "delhi":
        data = load_delhi_data()
        task = "cloud"
        model_path = os.path.join(MODELS_DIR, "model_cloud_int8.tflite")
        rgb_all = data["rgb"]
        thm_all = data["thm"]
        labels = data["y_cloud"]
        class_names = {0: "CLEAR SKY / GROUND", 1: "CLOUD COVER DETECTED"}
        colors = {0: "#4ade80", 1: "#38bdf8"}
    else:
        data = load_siberia_fire_data()
        task = "fire"
        model_path = os.path.join(MODELS_DIR, "model_fire_int8.tflite")
        rgb_all = data["rgb"]
        thm_all = data["thm"]
        labels = data["y_fire"]
        class_names = {0: "NOMINAL FOREST / NO FIRE", 1: "WILDFIRE THERMAL ANOMALY"}
        colors = {0: "#4ade80", 1: "#ef4444"}

    engine = EdgeInferenceEngine(model_path=model_path)
    N = len(labels)

    # Normalize thermal
    thm_valid = thm_all[~np.isnan(thm_all)]
    mu_t = float(np.mean(thm_valid)) if len(thm_valid) > 0 else 0.0
    sd_t = float(np.std(thm_valid)) + 1e-6 if len(thm_valid) > 0 else 1.0

    thm_norm = np.nan_to_num((thm_all - mu_t) / sd_t, nan=0.0)
    if thm_norm.ndim == 3:
        thm_norm = np.expand_dims(thm_norm, -1)

    # Setup Matplotlib UI
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 8.5))
    fig.patch.set_facecolor("#0b0f19")

    controller = VisualizerController(initial_idx=sample_idx % N, total_samples=N, is_slideshow=slideshow)
    fig.canvas.mpl_connect("key_press_event", controller.on_key)
    fig.canvas.mpl_connect("close_event", controller.on_close)

    if slideshow:
        print(f"\n=======================================================")
        print(f" Slideshow Mode Active ({delay_s:.1f}s per pass)")
        print(f" Controls: [Space] Pause/Resume | [->] Next | [<-] Prev | [Q] Quit")
        print(f"=======================================================\n")

    while controller.running:
        idx = controller.idx
        plt.clf()

        rgb_raw = rgb_all[idx]
        thm_raw = thm_all[idx]
        y_true = labels[idx]

        # Prepare model inputs
        x_rgb = rgb_raw.astype(np.float32) / 255.0
        x_thm = thm_norm[idx].astype(np.float32)

        # Run Edge AI Inference
        t0 = time.perf_counter()
        pred_class, confidence = engine.predict_sample(x_rgb, x_thm)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Create Fused Overlay
        thm_clean = np.nan_to_num(thm_raw, nan=mu_t)
        thm_min, thm_max = np.min(thm_clean), np.max(thm_clean)
        thm_scaled = (thm_clean - thm_min) / (thm_max - thm_min + 1e-6)
        thm_resized = cv2.resize(thm_scaled, (128, 128), interpolation=cv2.INTER_LINEAR)
        thm_heat = cv2.applyColorMap((thm_resized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        thm_heat = cv2.cvtColor(thm_heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        fused_img = 0.55 * x_rgb + 0.45 * thm_heat
        fused_img = np.clip(fused_img, 0.0, 1.0)

        # Header Title
        pause_status = " [PAUSED]" if controller.paused else ""
        fig.suptitle(
            f"EdgeAI Dual-Payload Satellite Live Pass  |  Sample #{idx:04d} / {N:04d}  |  Altitude: 500 km{pause_status}",
            fontsize=17, fontweight="bold", color="#f8fafc", y=0.96
        )

        # Panel 1: Optical RGB
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(rgb_raw)
        ax1.set_title("1. Optical Camera (RGB 128x128)\n[What Human Eyes See]", fontsize=12, color="#94a3b8", pad=8)
        ax1.axis("off")

        # Panel 2: Thermal Infrared
        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(thm_clean, cmap="inferno", aspect="auto")
        ax2.set_title("2. MLX90640 Thermal Sensor (24x32)\n[Native Radiometric Heat Vision]", fontsize=12, color="#94a3b8", pad=8)
        ax2.axis("off")
        cbar = plt.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.06, shrink=0.8)
        cbar.set_label("Brightness Temperature (Kelvin / DN)", color="#94a3b8", fontsize=10)

        # Panel 3: Fused Super-Vision
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(fused_img)
        ax3.set_title("3. Multi-Spectral Fused Overlay\n[Optical RGB + Thermal Heat Blend]", fontsize=12, color="#94a3b8", pad=8)
        ax3.axis("off")

        # AI Decision Badge
        pred_label_str = class_names.get(pred_class, f"CLASS {pred_class}")
        true_label_str = class_names.get(y_true, f"CLASS {y_true}")
        badge_col = colors.get(pred_class, "#38bdf8")
        is_correct = (pred_class == y_true)
        status_icon = "CORRECT" if is_correct else "MISMATCH"

        info_box = (
            f"  >> ONBOARD EDGE AI DECISION :  {pred_label_str} ({status_icon})\n"
            f"  ----------------------------------------------------------------------------------------------------\n"
            f"  * Ground Truth Label   : {true_label_str}\n"
            f"  * AI Prediction Prob   : {confidence * 100:.1f}%\n"
            f"  * Edge Inference Time  : {latency_ms:.2f} ms (Target: < 100 ms)\n"
            f"  * Model Architecture   : Two-Branch Late Fusion CNN (Int8 Quantized: 43.7 KB)\n"
            f"  * Sensor Payloads      : Sony IMX477 (RGB) + Melexis MLX90640 (24x32 Thermal)\n"
            f"  * Slideshow Controls   : [Space] Pause/Resume | [->] Next | [<-] Prev | [Q] Quit"
        )

        fig.text(
            0.5, 0.08, info_box,
            fontsize=10.5, family="monospace", color="#f8fafc",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#1e293b", edgecolor=badge_col, linewidth=2.5)
        )

        plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.22, wspace=0.15)

        if save_fig:
            out_name = f"demo_{scene}_sample_{idx:04d}.png"
            plt.savefig(out_name, dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved visualization to {out_name}")
            return

        if not slideshow:
            plt.show()
            break

        # Slideshow step management with responsive pause & key handling
        controller.manual_step = False
        t_waited = 0.0
        step_dt = 0.1
        while controller.running and not controller.manual_step:
            plt.pause(step_dt)
            if not controller.paused:
                t_waited += step_dt
                if t_waited >= delay_s:
                    controller.idx = (controller.idx + 1) % N
                    break


def main():
    parser = argparse.ArgumentParser(description="EdgeAI Satellite Interactive Demo Visualizer.")
    parser.add_argument("--scene", type=str, default="siberia", choices=["delhi", "siberia"], help="Scene to demonstrate")
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--slideshow", action="store_true", help="Run automated slideshow pass")
    parser.add_argument("--delay", type=float, default=5.0, help="Slideshow transition delay in seconds (default: 5.0s)")
    parser.add_argument("--save", action="store_true", help="Save visualization to PNG")
    args = parser.parse_args()

    try:
        run_visual_demo(
            scene=args.scene,
            sample_idx=args.sample,
            save_fig=args.save,
            slideshow=args.slideshow,
            delay_s=args.delay
        )
    except KeyboardInterrupt:
        print("\nSlideshow closed cleanly.")
        plt.close("all")


if __name__ == "__main__":
    main()
