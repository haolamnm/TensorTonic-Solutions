import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions = np.asarray(prob_distributions, dtype=float)
    actual_tokens = np.asarray(actual_tokens, dtype=int)

    N = len(actual_tokens)
    probs = prob_distributions[np.arange(N), actual_tokens]

    # Remember to handle log(0)
    safe_probs = np.clip(probs, 1e-12, 1.0)

    entropy = - np.mean(np.log(safe_probs))
    perplexity = np.exp(entropy)
    
    return perplexity