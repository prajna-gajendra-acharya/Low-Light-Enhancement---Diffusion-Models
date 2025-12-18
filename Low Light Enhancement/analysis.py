"""
===========================================================
 ANALYSIS VISUALIZATION SCRIPT
 - Triplet (Low | Enhanced | GT)
 - Error Heatmap
 - Local SSIM Map
 - Brightness Histograms
 - Error Distribution
===========================================================
RUN THIS ONLY AFTER evaluate.py has produced restored images
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF


plt.style.use("seaborn-v0_8-whitegrid")
to_t = T.ToTensor()

# --------------------------------------------------------------
# USER — UPDATE THIS ROOT TO YOUR LOL-v1 FOLDER
# --------------------------------------------------------------

ROOT = Path("/content/drive/MyDrive/CV_Project/LightenDiffusion/LOL-v1")

LOW_DIR  = ROOT / "val" / "low"
HIGH_DIR = ROOT / "val" / "high"
PRED_DIR = ROOT / "LOLv1"       # evaluate.py saves restored images here


# --------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------

def load_triplet(img_id):
    """Loads low-light, enhanced, and ground-truth images as same resolution."""
    low_path  = LOW_DIR  / img_id
    gt_path   = HIGH_DIR / img_id
    pred_path = PRED_DIR / img_id  # restored name must match

    # Load as tensors
    low  = to_t(Image.open(low_path).convert("RGB"))
    gt   = to_t(Image.open(gt_path).convert("RGB"))
    pred = to_t(Image.open(pred_path).convert("RGB"))

    # gt defines the reference resolution
    _, H, W = gt.shape

    # Resize low & pred to match GT size
    low_resized  = TF.resize(low,  [H, W], antialias=True)
    pred_resized = TF.resize(pred, [H, W], antialias=True)

    return low_resized, pred_resized, gt



def to_np_img(t):
    t = t.detach().cpu().clamp(0, 1).numpy()
    return np.transpose(t, (1, 2, 0))  # CHW → HWC


def to_gray(t):
    """Convert tensor image to grayscale numpy array [H,W]."""
    t = t.detach().cpu().clamp(0, 1).numpy()
    t = np.transpose(t, (1, 2, 0))
    return (0.299 * t[..., 0] +
            0.587 * t[..., 1] +
            0.114 * t[..., 2])


# --------------------------------------------------------------
# PLOT 1 — Full Triplet + Heatmap + SSIM Map
# --------------------------------------------------------------

def plot_triplet_error_ssim(img_id, save=True):
    low_t, pred_t, gt_t = load_triplet(img_id)

    low_np  = to_np_img(low_t)
    pred_np = to_np_img(pred_t)
    gt_np   = to_np_img(gt_t)

    # Error map
    err_map = np.mean(np.abs(pred_np - gt_np), axis=-1)

    # SSIM map
    pred_gray = to_gray(pred_t)
    gt_gray   = to_gray(gt_t)
    _, ssim_map = ssim(gt_gray, pred_gray, data_range=1.0, full=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0,0].imshow(low_np)
    axes[0,0].set_title("Low-Light Input")
    axes[0,0].axis("off")

    axes[0,1].imshow(pred_np)
    axes[0,1].set_title("Enhanced Output")
    axes[0,1].axis("off")

    axes[0,2].imshow(gt_np)
    axes[0,2].set_title("Ground Truth")
    axes[0,2].axis("off")

    im0 = axes[1,0].imshow(err_map, cmap="magma")
    axes[1,0].axis("off")
    fig.colorbar(im0, ax=axes[1,0])

    im1 = axes[1,1].imshow(ssim_map, cmap="viridis", vmin=0, vmax=1)
    axes[1,1].axis("off")
    fig.colorbar(im1, ax=axes[1,1])

    axes[1,2].axis("off")
    axes[1,2].text(
        0.05, 0.95,
        f"Image: {img_id}\n\n"
        "Heatmap → where model deviates from GT\n"
        "SSIM → structural similarity map\n"
        "Bright = high similarity\n"
        "Dark = mismatched structure",
        fontsize=12,
        va="top",
    )

    plt.tight_layout()

    if save:
        out_path = ROOT / f"{img_id.replace('.png','')}_analysis.png"
        plt.savefig(out_path, dpi=220)
        print("Saved:", out_path)

    plt.show()


# --------------------------------------------------------------
# PLOT 2 — Histograms
# --------------------------------------------------------------

def plot_histograms(img_id, save=True):
    low_t, pred_t, gt_t = load_triplet(img_id)

    low_gray  = to_gray(low_t).ravel()
    pred_gray = to_gray(pred_t).ravel()
    gt_gray   = to_gray(gt_t).ravel()
    err       = np.abs(pred_gray - gt_gray)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    bins = np.linspace(0, 1, 50)

    # Brightness distributions
    ax[0].hist(low_gray, bins=bins, alpha=0.5, label="Low-Light", color="#1f77b4")
    ax[0].hist(pred_gray, bins=bins, alpha=0.5, label="Enhanced", color="#ff7f0e")
    ax[0].hist(gt_gray,   bins=bins, alpha=0.5, label="Ground Truth", color="#2ca02c")
    ax[0].set_title("Brightness Histogram")
    ax[0].set_xlabel("Intensity")
    ax[0].set_ylabel("Pixel Count")
    ax[0].legend()

    # Error distribution
    ax[1].hist(err, bins=40, alpha=0.8, color="#d62728")
    ax[1].set_title("Error Distribution |pred - gt|")
    ax[1].set_xlabel("Absolute Error")
    ax[1].set_ylabel("Pixel Count")


    if save:
        out_path = ROOT / f"{img_id.replace('.png','')}_histograms.png"
        plt.savefig(out_path, dpi=220)
        print("Saved:", out_path)

    plt.show()


# --------------------------------------------------------------
# MAIN ENTRY — Run analysis
# --------------------------------------------------------------

def analyze(img_id):
    print("\n==========================")
    print("Analyzing", img_id)
    print("==========================\n")

    plot_triplet_error_ssim(img_id)
    plot_histograms(img_id)


if __name__ == "__main__":
    # Modify which images you want to analyze
    analyze("111.png")
    analyze("79.png")
