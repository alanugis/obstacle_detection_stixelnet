import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
import tqdm
from albumentations import Compose, Normalize
import matplotlib.pyplot as plt

from config import Config
from data_loader import KittiStixelDataset
from models import build_stixel_net

def compute_auc(eps_values, f_values):
    integral = np.trapz(f_values, eps_values)
    return integral / 50.0

def evaluate_model():
    dt_config = Config()
    val_aug = Compose([Normalize(p=1.0)])
    val_set = KittiStixelDataset(
        data_path=dt_config.DATA_PATH,
        ground_truth_path=dt_config.GROUND_TRUTH_PATH,
        phase="val",
        batch_size=8,
        transform=val_aug,
        shuffle=False
    )
    
    print("Loading model...")
    model = build_stixel_net()
    model.load_weights(os.path.join("saved_models", "model-002.h5"))
    
    all_abs_errors = []
    
    eps_values = np.linspace(0, 50, 100)
    prob_sums = {eps: [] for eps in eps_values}
    
    print("Evaluating over validation dataset...")
    for X, y_true in tqdm.tqdm(val_set, total=len(val_set)):
        preds = model.predict(X, verbose=0)
        preds = np.reshape(preds, (X.shape[0], 100, 50))
        
        preds_exp = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
        probs = preds_exp / np.sum(preds_exp, axis=-1, keepdims=True)
        
        y_pred_idx = np.argmax(probs, axis=-1)
        
        for b in range(X.shape[0]):
            for col in range(100):
                have_gt = y_true[b, col, 0]
                if have_gt == 1:
                    gt_y = y_true[b, col, 1]
                    pred_y = y_pred_idx[b, col]
                    
                    gt_y_px = gt_y * (370.0 / 50.0)
                    pred_y_px = pred_y * (370.0 / 50.0)
                    abs_error_px = np.abs(gt_y_px - pred_y_px)
                    
                    all_abs_errors.append(abs_error_px)
                    
                    bin_px = np.arange(50) * (370.0 / 50.0)
                    dist_px = np.abs(bin_px - gt_y_px)
                    col_probs = probs[b, col, :]
                    
                    for eps in eps_values:
                        mask = dist_px < eps
                        prob_mass = np.sum(col_probs[mask])
                        prob_sums[eps].append(prob_mass)

    all_abs_errors = np.array(all_abs_errors)
    print(f"Total valid samples evaluated: {len(all_abs_errors)}")
    
    f_values = []
    for eps in eps_values:
        f_values.append(np.mean(all_abs_errors < eps))
        
    f_values = np.array(f_values)
    auc = compute_auc(eps_values, f_values)
    
    prob_mass_avg = []
    for eps in eps_values:
        prob_mass_avg.append(np.mean(prob_sums[eps]))
    prob_mass_avg = np.array(prob_mass_avg)
    
    print(f"Final StixelNet AUC: {auc:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 6(a)
    ax = axes[0, 0]
    ax.plot(eps_values, f_values, label=f"StixelNet (AUC={auc:.3f})", color='b')
    ax.set_title("Figure 6(a): Compared Methods")
    ax.set_xlabel(r"$\epsilon$ (pixels)")
    ax.set_ylabel(r"Fraction of samples with error < $\epsilon$")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")

    # 6(b)
    ax = axes[0, 1]
    ax.plot(eps_values, f_values, label=f"StixelNet (AUC={auc:.3f})", color='r')
    ax.set_title("Figure 6(b): Different Loss Functions")
    ax.set_xlabel(r"$\epsilon$ (pixels)")
    ax.set_ylabel(r"Fraction of samples with error < $\epsilon$")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")
    
    # 6(c)
    ax = axes[1, 0]
    ax.plot(eps_values, prob_mass_avg, label="StixelNet", color='g')
    ax.set_title("Figure 6(c): Average Probability Mass")
    ax.set_xlabel(r"$\epsilon$ (pixels)")
    ax.set_ylabel("Average probability mass near GT")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")
    
    # 6(d)
    ax = axes[1, 1]
    ax.plot(eps_values, f_values, label=f"StixelNet w/o CRF (AUC={auc:.3f})", color='m')
    ax.set_title("Figure 6(d): StixelNet with and without CRF")
    ax.set_xlabel(r"$\epsilon$ (pixels)")
    ax.set_ylabel(r"Fraction of samples with error < $\epsilon$")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("evaluate_stixelnet_results.png", dpi=300)
    print("Saved plot to evaluate_stixelnet_results.png")
    
if __name__ == '__main__':
    evaluate_model()
