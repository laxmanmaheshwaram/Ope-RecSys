import numpy as np

class RecSysEnv:
    def __init__(self, n_arms=5, context_dim=10):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.true_theta = np.random.uniform(-1, 1, (n_arms, context_dim))

    def get_context(self):
        return np.random.normal(0, 1, self.context_dim)

    def get_reward(self, context, arm):
        return np.dot(context, self.true_theta[arm]) + np.random.normal(0, 0.1)