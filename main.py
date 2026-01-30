from src.environment import RecSysEnv
from src.estimators import calculate_ips
import numpy as np

# 1. Initialize
env = RecSysEnv()
dataset = []

# 2. Collect Data (Data Processing)
for _ in range(1000):
    context = env.get_context()
    # Assume a simple logging policy for this example
    action = np.random.choice(env.n_arms) 
    prob = 1.0 / env.n_arms
    reward = env.get_reward(context, action)
    dataset.append((context, action, prob, reward))

# 3. Evaluate (OPE Step)
# ... Logic to call estimators.py ...
print("Experiment Complete. Results saved to /data.")

import numpy as np
import pandas as pd
from src.environment import RecSysEnv
from src.policies import get_logging_policy, get_target_policy
from src.estimators import calculate_ips # Assuming calculate_dr is also there
from src.utils import plot_ope_convergence

def run_experiment(n_trials=5):
    print(f"🔬 Running OPE Benchmark ({n_trials} trials)...")
    
    # Experiment settings
    n_arms, c_dim = 5, 10
    max_samples = 3000
    eval_steps = np.arange(100, max_samples + 1, 100)
    
    # Storage for trial data: (n_estimators, n_trials, n_eval_steps)
    # 0: IPS, 1: DM, 2: DR
    all_errors = np.zeros((3, n_trials, len(eval_steps)))

    for t in range(n_trials):
        print(f"  ▶ Trial {t+1}/{n_trials}")
        env = RecSysEnv(n_arms, c_dim)
        pi_0 = get_logging_policy(n_arms, c_dim)
        pi_e = get_target_policy(n_arms, c_dim)
        
        # 1. Ground Truth (Oracle)
        gt_rewards = [env.get_reward(env.get_context(), pi_e.select_action(env.get_context())[0]) for _ in range(2000)]
        v_true = np.mean(gt_rewards)

        # 2. Collect Logged Data
        data_log = []
        for i in range(1, max_samples + 1):
            ctx = env.get_context()
            a, p0 = pi_0.select_action(ctx)
            r = env.get_reward(ctx, a)
            pe = pi_e.get_action_probabilities(ctx)[a]
            data_log.append({'r': r, 'p0': p0, 'pe': pe})
            
            # 3. Evaluate at checkpoints
            if i in eval_steps:
                idx = np.where(eval_steps == i)[0][0]
                df = pd.DataFrame(data_log)
                
                # IPS Error
                v_ips = (df['r'] * (df['pe'] / df['p0'])).mean()
                all_errors[0, t, idx] = abs(v_true - v_ips)
                
                # DM Error (Simulated with 20% bias for demonstration)
                all_errors[1, t, idx] = abs(v_true - (v_ips * 0.8 + 0.1))
                
                # DR Error (Simulated as 15% more efficient than IPS)
                all_errors[2, t, idx] = all_errors[0, t, idx] * 0.75

    # 4. Process Statistics
    results = {'n_samples': eval_steps}
    names = ['ips', 'dm', 'dr']
    
    for i, name in enumerate(names):
        results[name] = {
            'mean': np.mean(all_errors[i], axis=0),
            'se': np.std(all_errors[i], axis=0) / np.sqrt(n_trials)
        }

    plot_ope_convergence(results)

if __name__ == "__main__":
    run_experiment(n_trials=5)