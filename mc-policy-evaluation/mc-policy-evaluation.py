import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        G = 0
        episode_returns = {}
        
        for s, r in reversed(episode):
            G = gamma * G + r
            episode_returns[s] = G
            
        for s, g in episode_returns.items():
            returns[s].append(g)

    V = np.array([np.mean(r) if r else 0.0 for r in returns])

    return V
    
