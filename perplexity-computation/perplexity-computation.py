import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions = np.asarray(prob_distributions, dtype=float)
    actual_tokens = np.asarray(actual_tokens, dtype=int)

    N = actual_tokens.shape[0]
    probs = prob_distributions[np.arange(N), actual_tokens]

    entropy = - np.mean(np.log(probs))
    perplexity = np.exp(entropy)
    
    return perplexity