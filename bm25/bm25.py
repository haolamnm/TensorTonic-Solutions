import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.array([])
        
    # 1. calculate document lengths and average document length (avgdl)
    doc_lengths = [len(doc) for doc in docs]
    avgdl = sum(doc_lengths) / N
    
    # 2. calculate (df) and (IDF)
    # only need to compute this for the terms present in the query
    idf = {}
    unique_query_terms = set(query_tokens)
    
    for term in unique_query_terms:
        # count how many documents contain the term
        df_t = sum(1 for doc in docs if term in doc)
        
        # IDF formula as defined in the prompt
        idf[term] = math.log(((N - df_t + 0.5) / (df_t + 0.5)) + 1)
        
    # 3. calculate the BM25 score for each document
    scores = np.zeros(N)
    
    for i, doc in enumerate(docs):
        D_len = doc_lengths[i]
        doc_term_counts = Counter(doc)
        
        doc_score = 0
        for term in unique_query_terms:
            tf_t_D = doc_term_counts.get(term, 0)
            
            if tf_t_D == 0:
                continue # term not in document, contributes 0 to the sum
                
            # BM25 Term Score Formula
            numerator = tf_t_D * (k1 + 1)
            denominator = tf_t_D + k1 * (1 - b + b * (D_len / avgdl))
            
            doc_score += idf[term] * (numerator / denominator)
            
        scores[i] = doc_score
        
    return scores