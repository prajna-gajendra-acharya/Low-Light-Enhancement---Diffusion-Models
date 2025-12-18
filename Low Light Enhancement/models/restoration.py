import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

import utils

from skimage.metrics import structural_similarity as ssim_fn
import lpips


def tensor_to_uint8(img):
    """
    img: (B,3,H,W) or (3,H,W), float in [0,1] (assumed).
    Returns numpy uint8 in HxWx3.
    """
    if img.dim() == 4:
        img = img[0]
    img = img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return img


def compute_psnr(pred, gt):
    """
    pred, gt: (B,3,H,W) float in [0,1].
    """
    mse = torch.mean((pred - gt) ** 2)
    if mse.item() == 0:
        return 100.0
    psnr = -10.0 * torch.log10(mse)
    return psnr.item()


def compute_ssim(pred, gt):
    """
    pred, gt: (B,3,H,W) float in [0,1], returns scalar SSIM.
    Uses skimage.structural_similarity on RGB.
    """
    pred_np = tensor_to_uint8(pred)
    gt_np = tensor_to_uint8(gt)
    # channel_axis=2 for (H,W,C)
    ssim_val = ssim_fn(pred_np, gt_np, data_range=255, channel_axis=2)
    return float(ssim_val)


def compute_lpips(pred, gt, lpips_model):
    """
    pred, gt: (B,3,H,W) float in [0,1].
    lpips_model: lpips.LPIPS instance on correct device.
    """
    # LPIPS expects normalized [-1,1]
    pred_n = (pred * 2.0 - 1.0)
    gt_n = (gt * 2.0 - 1.0)
    d = lpips_model(pred_n, gt_n)
    return float(d.mean().item())


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        # Load stage-2 diffusion checkpoint
        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained model path is missing!')

        # Set up LPIPS for evaluation
        print("Setting up LPIPS (VGG) for evaluation...")
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.config.device)
        self.lpips_model.eval()

        # Where to save numeric metrics
        # e.g. <data_dir>/LOLv1/eval_metrics.txt
        self.metrics_path = os.path.join(
            self.config.data.data_dir,
            self.config.data.val_dataset,
            "eval_metrics.txt"
        )
        os.makedirs(
            os.path.dirname(self.metrics_path),
            exist_ok=True
        )

        # If file doesn't exist yet, write header
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write("img_id,psnr,ssim,lpips\n")

    def restore(self, val_loader):
        """
        Runs restoration on validation loader, computes metrics vs. GT,
        and also saves Low | Enhanced | GT triplet images.
        """
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        os.makedirs(image_folder, exist_ok=True)

        self.diffusion.model.eval()

        all_psnr, all_ssim, all_lpips = [], [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                # x has shape (B, 6, H, W) because dataset returns [low, high] concatenated
                # Split into low and GT
                low = x[:, :3, :, :].to(self.diffusion.device)
                gt  = x[:, 3:, :, :].to(self.diffusion.device)

                b, c, h, w = low.shape

                # Pad low to multiple of 64 like in training
                img_h_64 = int(64 * np.ceil(h / 64.0))
                img_w_64 = int(64 * np.ceil(w / 64.0))
                x_cond = F.pad(low, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')

                t1 = time.time()
                # Match training-style call: concatenate condition twice to get 6 channels
                pred_x = self.diffusion.model(
                    torch.cat((x_cond, x_cond), dim=1)
                )["pred_x"][:, :, :h, :w]
                t2 = time.time()

                # --- Metrics ---
                # Assume pred_x and gt are already in [0,1]-ish
                psnr_val = compute_psnr(pred_x, gt)
                ssim_val = compute_ssim(pred_x, gt)
                lpips_val = compute_lpips(pred_x, gt, self.lpips_model)

                all_psnr.append(psnr_val)
                all_ssim.append(ssim_val)
                all_lpips.append(lpips_val)

                img_id = y[0] if isinstance(y, (list, tuple)) else str(y)

                # Log per-image metrics to console
                print(f"Image {img_id}: "
                      f"PSNR={psnr_val:.3f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}, "
                      f"time={t2 - t1:.4f}s")

                # Append metrics to text file
                with open(self.metrics_path, "a") as f:
                    f.write(f"{img_id},{psnr_val:.6f},{ssim_val:.6f},{lpips_val:.6f}\n")

                # --- Save enhanced image (like original code) ---
                utils.logging.save_image(
                    pred_x,
                    os.path.join(image_folder, f"{img_id}")
                )

                # --- Save Low | Enhanced | GT triplet as a grid ---
                triplet = torch.cat([low.cpu(), pred_x.cpu(), gt.cpu()], dim=0)
                grid = make_grid(triplet, nrow=3)
                grid_path = os.path.join(image_folder, f"{img_id}_triplet.png")
                save_image(grid, grid_path)

            # Optional: print average metrics at the end
            if len(all_psnr) > 0:
                print("====== Evaluation Summary ======")
                print(f"Avg PSNR : {np.mean(all_psnr):.3f}")
                print(f"Avg SSIM : {np.mean(all_ssim):.4f}")
                print(f"Avg LPIPS: {np.mean(all_lpips):.4f}")
                print(f"Metrics saved to: {self.metrics_path}")
