import matplotlib.pyplot as plt
import numpy as np

def plot_ope_convergence(results_df):
    """
    Plots the Mean Absolute Error (MAE) of different OPE estimators 
    relative to the Ground Truth as sample size increases.
    """
    plt.figure(figsize=(10, 6))
    
    # Plotting each estimator's performance
    plt.plot(results_df['n_samples'], results_df['ips_error'], label='IPS (Inverse Propensity)', color='blue', linestyle='--')
    plt.plot(results_df['n_samples'], results_df['dm_error'], label='DM (Direct Method)', color='red', linestyle=':')
    plt.plot(results_df['n_samples'], results_df['dr_error'], label='DR (Doubly Robust)', color='green', linewidth=2)

    plt.title('OPE Convergence: Estimator Error vs. Sample Size', fontsize=14)
    plt.xlabel('Number of Logged Samples (n)', fontsize=12)
    plt.ylabel('Absolute Error from Ground Truth', fontsize=12)
    plt.xscale('log')  # Log scale is better for seeing convergence
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Save the figure to the project's root for the README
    plt.savefig('results_plot.png')
    print("Plot saved as results_plot.png")
    plt.show()