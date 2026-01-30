import numpy as np
import pandas as pd
from src.environment import RecSysEnv
from src.policies import RecommendationPolicy, softmax
from src.estimators import calculate_ips, calculate_dr

def run_robustness():
    env = RecSysEnv(n_arms=5, context_dim=10)
    # Logging policy (pi_0) is fixed and exploratory
    pi_0 = RecommendationPolicy(5, 10, temperature=2.0)
    
    # We vary the target policy temperature from 'similar' to 'very different'
    temperatures = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2]
    results = []

    print("Running Robustness Study: Varying Target Policy Temperature...")

    for temp in temperatures:
        pi_e = RecommendationPolicy(5, 10, temperature=temp)
        
        # Collect Logged Data (n=2000)
        ips_errors = []
        for trial in range(10): # Multiple trials for error bars
            contexts = [env.get_context() for _ in range(2000)]
            logged_data = [pi_0.select_action(c) for c in contexts]
            rewards = [env.get_reward(c, a) for c, (a, p) in zip(contexts, logged_data)]
            
            # Get Target Probabilities for the same actions
            target_probs = [pi_e.get_action_probabilities(c)[a] for c, (a, p) in zip(contexts, logged_data)]
            logged_probs = [p for (a, p) in logged_data]
            
            # Ground Truth (Monte Carlo estimate of pi_e)
            gt_reward = np.mean([np.dot(c, env.true_theta[pi_e.select_action(c)[0]]) for c in contexts])
            
            # Estimates
            est_ips = calculate_ips(np.array(target_probs), np.array(logged_probs), np.array(rewards))
            ips_errors.append(np.abs(est_ips - gt_reward))

        results.append({
            'temperature': temp,
            'ips_mae': np.mean(ips_errors),
            'ips_std': np.std(ips_errors)
        })

    # Save for plotting
    df = pd.DataFrame(results)
    df.to_csv("results/robustness_results.csv", index=False)
    print("Done. Results saved to results/robustness_results.csv")

if __name__ == "__main__":
    run_robustness()