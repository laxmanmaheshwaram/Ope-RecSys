import numpy as np
from src.environment import RecSysEnv
from src.policies import get_logging_policy, get_target_policy
from src.estimators import calculate_dr

def run_sensitivity():
    env = RecSysEnv(n_arms=5, context_dim=10)
    pi_0 = get_logging_policy(5, 10)
    pi_e = get_target_policy(5, 10)
    
    # Generate data
    contexts = [env.get_context() for _ in range(1000)]
    actions, logged_probs = zip(*[pi_0.select_action(c) for c in contexts])
    rewards = np.array([env.get_reward(c, a) for c, a in zip(contexts, actions)])
    
    # Research Variable: Noise in the Reward Model
    # A 'perfect' researcher tests what happens when their model is WRONG.
    noise_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    for noise in noise_levels:
        # Simulate a Reward Model (Direct Method) with varying error
        predicted_rewards = rewards + np.random.normal(0, noise, size=rewards.shape)
        
        target_probs = np.array([pi_e.get_action_probabilities(c)[a] for c, a in zip(contexts, actions)])
        
        dr_val = calculate_dr(target_probs, np.array(logged_probs), rewards, predicted_rewards, actions)
        print(f"Reward Model Noise: {noise} | DR Estimate: {dr_val:.4f}")

if __name__ == "__main__":
    run_sensitivity()