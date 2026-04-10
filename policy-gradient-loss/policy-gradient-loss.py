import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    T = len(rewards)
    returns = np.zeros(T)
    
    # G_t = r_t + gamma * G_t+1
    current_return = 0
    for t in reversed(range(T)):
        current_return = rewards[t] + gamma * current_return
        returns[t] = current_return
    
    mean_return = np.mean(returns)
    advantages = returns - mean_return
    
    # L = -1/T * sum(log_pi * Advantage)
    loss = -np.mean(log_probs * advantages)
    
    return loss