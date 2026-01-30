import numpy as np

def softmax(x, temperature=1.0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum(axis=0)

class RecommendationPolicy:
    def __init__(self, n_arms, context_dim, temperature=1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.temperature = temperature
        # In a real scenario, this 'weights' matrix would be learned from data
        # Here we use a fixed matrix to represent a specific behavior
        self.weights = np.random.uniform(-1, 1, (n_arms, context_dim))

    def get_action_probabilities(self, context):
        """
        Returns the probability distribution over actions for a given context.
        pi(a|x)
        """
        scores = np.dot(self.weights, context)
        probs = softmax(scores, self.temperature)
        return probs

    def select_action(self, context):
        """
        Samples an action based on the probability distribution.
        """
        probs = self.get_action_probabilities(context)
        action = np.random.choice(self.n_arms, p=probs)
        return action, probs[action]

# --- PhD Portfolio Tip: Explicitly define the two policies ---

def get_logging_policy(n_arms, context_dim):
    """
    pi_0: High temperature (e.g., 2.0) creates a more 'uniform' 
    distribution, representing exploration.
    """
    return RecommendationPolicy(n_arms, context_dim, temperature=2.0)

def get_target_policy(n_arms, context_dim):
    """
    pi_e: Low temperature (e.g., 0.5) creates a 'peakier' 
    distribution, representing a more confident, greedy model.
    """
    return RecommendationPolicy(n_arms, context_dim, temperature=0.5)