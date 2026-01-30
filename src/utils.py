import matplotlib.pyplot as plt
import numpy as np

def plot_ope_convergence(results_dict):
    """
    Plots the Mean Absolute Error (MAE) with Standard Error shading.
    results_dict expects: {
        'n_samples': np.array,
        'ips': {'mean': np.array, 'se': np.array},
        'dm': {'mean': np.array, 'se': np.array},
        'dr': {'mean': np.array, 'se': np.array}
    }
    """
    plt.figure(figsize=(12, 7))
    n = results_dict['n_samples']
    
    # Estimator configurations for plotting
    configs = [
        ('ips', 'IPS (Inverse Propensity)', 'blue', '--'),
        ('dm', 'DM (Direct Method)', 'red', ':'),
        ('dr', 'DR (Doubly Robust)', 'green', '-')
    ]
    
    for key, label, color, style in configs:
        mean = results_dict[key]['mean']
        se = results_dict[key]['se']
        
        # Plot the main mean line
        plt.plot(n, mean, label=label, color=color, linestyle=style, linewidth=2)
        
        # Add the Shaded Confidence Interval (Standard Error)
        plt.fill_between(n, mean - se, mean + se, color=color, alpha=0.15)

    plt.title('OPE Convergence: Estimator Error vs. Sample Size (with Standard Error)', fontsize=14)
    plt.xlabel('Number of Logged Samples (n)', fontsize=12)
    plt.ylabel('Absolute Error from Ground Truth', fontsize=12)
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    plt.savefig('results/ope_research_plot.png')
    #print(" Plot saved as ope_research_plot.png")
    plt.show()