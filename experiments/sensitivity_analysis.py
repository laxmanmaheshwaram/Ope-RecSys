import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.environment import RecSysEnv
from src.policies import get_logging_policy, get_target_policy
from src.estimators import calculate_dr, calculate_ips

def run_sensitivity_study():
    # Setup Research Environment
    env = RecSysEnv(n_arms=10, context_dim=5)
    pi_0 = get_logging_policy(10, 5) # Exploratory
    pi_e = get_target_policy(10, 5)  # Exploitative
    
    # 1. Generate 'Logged Data' (The Evidence)
    n_samples = 5000
    contexts = [env.get_context() for _ in range(n_samples)]
    actions, logged_probs = zip(*[pi_0.select_action(c) for c in contexts])
    rewards = np.array([env.get_reward(c, a) for c, a in zip(contexts, actions)])
    
    # 2. Ground Truth (The "Perfect" Oracle for comparison)
    gt_reward = np.mean([np.dot(c, env.true_theta[pi_e.select_action(c)[0]]) for c in contexts])

    # 3. Sensitivity Sweep: Varying Reward Model Accuracy
    # We simulate a "Direct Method" (DM) model that gets progressively worse
    noise_levels = np.linspace(0, 5, 10) 
    results = []

    for noise in noise_levels:
        # Simulate predicted rewards with Gaussian noise
        # As noise increases, the "Double Robustness" is tested
        predicted_rewards = rewards + np.random.normal(0, noise, size=rewards.shape)
        
        target_probs = np.array([pi_e.get_action_probabilities(c)[a] for c, a in zip(contexts, actions)])
        
        # Calculate Estimators
        est_ips = calculate_ips(target_probs, np.array(logged_probs), rewards)
        est_dr = calculate_dr(target_probs, np.array(logged_probs), rewards, predicted_rewards, actions)
        
        results.append({
            'noise': noise,
            'ips_error': np.abs(est_ips - gt_reward),
            'dr_error': np.abs(est_dr - gt_reward)
        })

    # 4. Save and Visualize
    df = pd.DataFrame(results)
    plot_sensitivity(df)
    df.to_csv("results/sensitivity_results.csv", index=False)

def plot_sensitivity(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['noise'], df['dr_error'], label='Doubly Robust Error', marker='o')
    plt.axhline(y=df['ips_error'].mean(), color='r', linestyle='--', label='IPS Error (Baseline)')
    plt.xlabel('Reward Model Noise ($\sigma$)')
    plt.ylabel('Absolute Error from Ground Truth')
    plt.title('Sensitivity Analysis: Impact of Reward Model Noise on DR Estimator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/sensitivity_plot.png')

if __name__ == "__main__":
    run_sensitivity_study()