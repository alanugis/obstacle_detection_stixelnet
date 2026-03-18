import numpy as np
import matplotlib.pyplot as plt

def compute_auc(eps_values, f_values):
    """
    Computes the Area Under Curve (AUC) as described:
    numerical integral of f(eps) from eps=0 to 50, divided by 50.
    """
    integral = np.trapezoid(f_values, eps_values)
    return integral / 50.0

def plot_error_fraction_curve(ax, title, data_dict):
    """
    Plots the fraction of test samples whose absolute prediction error |y - y_hat| is below eps.
    data_dict: dict of method_name -> absolute_errors_array (1D array)
    """
    eps_values = np.linspace(0, 50, 500)
    
    for method, abs_errors in data_dict.items():
        f_values = np.array([np.mean(abs_errors < eps) for eps in eps_values])
        auc = compute_auc(eps_values, f_values)
        ax.plot(eps_values, f_values, label=f"{method} (AUC={auc:.3f})")
        
    ax.set_title(title)
    ax.set_xlabel(r"$\epsilon$ (pixels)")
    ax.set_ylabel(r"Fraction of samples with error < $\epsilon$")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")

def plot_probability_mass_curve(ax, title, prob_mass_data):
    """
    Plots the average predicted probability mass within ±eps pixels around the GT.
    prob_mass_data: dict of method_name -> function that given eps returns mean prob mass
    """
    eps_values = np.linspace(0, 50, 500)
    
    for method, prob_func in prob_mass_data.items():
        mass_values = np.array([prob_func(eps) for eps in eps_values])
        ax.plot(eps_values, mass_values, label=f"{method}")
        
    ax.set_title(title)
    ax.set_xlabel(r"$\epsilon$ (pixels)")
    ax.set_ylabel("Average probability mass near GT")
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower right")

def main():
    np.random.seed(42)
    n_samples = 10000

    # ==========================================
    # 1. Generate Mock Data for Figures
    # ==========================================
    # absolute errors for 6(a)
    stixelnet_errors = np.abs(np.random.laplace(loc=0, scale=3.0, size=n_samples))
    stereo_errors = np.abs(np.random.normal(loc=0, scale=8.0, size=n_samples))
    hog_svm_errors = np.abs(np.random.normal(loc=0, scale=12.0, size=n_samples))
    max_grad_errors = np.abs(np.random.uniform(low=0, high=50, size=n_samples))
    
    data_6a = {
        "StixelNet": stixelnet_errors,
        "Stereo Stixels": stereo_errors,
        "HOG+SVM": hog_svm_errors,
        "MaxGradient": max_grad_errors
    }

    # absolute errors for 6(b)
    pl_loss_errors = np.abs(np.random.laplace(loc=0, scale=3.0, size=n_samples))
    softmax_loss_errors = np.abs(np.random.laplace(loc=0, scale=5.0, size=n_samples))
    kl_loss_errors = np.abs(np.random.laplace(loc=0, scale=6.5, size=n_samples))
    l2_loss_errors = np.abs(np.random.normal(loc=0, scale=10.0, size=n_samples))
    
    data_6b = {
        "PL-loss": pl_loss_errors,
        "Softmax-loss": softmax_loss_errors,
        "KL-loss": kl_loss_errors,
        "L2-loss": l2_loss_errors
    }

    # probability mass functions for 6(c)
    # mock functions that rise to 1 asymptotically
    prob_6c = {
        "PL-loss": lambda eps: 1 - np.exp(-eps / 5.0),
        "Softmax-loss": lambda eps: 1 - np.exp(-eps / 8.0),
        "KL-loss": lambda eps: 1 - np.exp(-eps / 10.0),
        "L2-loss": lambda eps: 1 - np.exp(-eps / 15.0)
    }

    # absolute errors for 6(d)
    w_crf = np.abs(np.random.laplace(loc=0, scale=2.5, size=n_samples))
    wo_crf = np.abs(np.random.laplace(loc=0, scale=4.0, size=n_samples))
    
    data_6d = {
        "StixelNet w/ CRF": w_crf,
        "StixelNet w/o CRF": wo_crf
    }

    # ==========================================
    # 2. Plotting
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Figure 6(a)
    plot_error_fraction_curve(axes[0, 0], "Figure 6(a): Different Methods", data_6a)
    
    # Figure 6(b)
    plot_error_fraction_curve(axes[0, 1], "Figure 6(b): Different Loss Functions", data_6b)
    
    # Figure 6(c)
    plot_probability_mass_curve(axes[1, 0], "Figure 6(c): Average Probability Mass", prob_6c)
    
    # Figure 6(d)
    plot_error_fraction_curve(axes[1, 1], "Figure 6(d): StixelNet with and without CRF", data_6d)

    plt.tight_layout()
    plt.savefig("figure_6_reproduction.png", dpi=300)
    print("Saved plot to figure_6_reproduction.png")
    
if __name__ == "__main__":
    main()
