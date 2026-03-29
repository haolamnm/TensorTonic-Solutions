import numpy as np

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    c = len(candidate)
    r = len(reference)

    # forgot this
    if c == 0:
        return 0.0
    
    # Brevity penalty
    if c >= r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)
    
    log_precisions = []
    
    for n in range(1, max_n + 1):
        # Count candidate n-grams
        cand_ngrams = {}
        for i in range(len(candidate) - n + 1):
            ng = tuple(candidate[i:i+n])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1
        
        # Count reference n-grams
        ref_ngrams = {}
        for i in range(len(reference) - n + 1):
            ng = tuple(reference[i:i+n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1
        
        # Clipped counts
        clipped = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in cand_ngrams.items())
        total = sum(cand_ngrams.values())
        
        if total == 0 or clipped == 0:
            return 0.0
        
        log_precisions.append(np.log(clipped / total))
    
    # Geometric mean via mean of logs
    score = bp * np.exp(np.mean(log_precisions))
    return float(score)