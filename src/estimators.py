import numpy as np

def calculate_ips(target_probs, logged_probs, rewards):
    """
    Computes the Inverse Propensity Score estimate.
    weights = pi_e(a|x) / pi_0(a|x)
    """
    weights = target_probs / logged_probs
    return np.mean(weights * rewards)

def calculate_dr(target_probs, logged_probs, rewards, predicted_rewards, action_taken):
    """
    Doubly Robust Estimator
    Combines the Reward Model and the Propensity Weights.
    """
    weights = target_probs / logged_probs
    # DR formula: DM_estimate + Weight * (Error)
    dr_values = predicted_rewards + weights * (rewards - predicted_rewards)
    return np.mean(dr_values)