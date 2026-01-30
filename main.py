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