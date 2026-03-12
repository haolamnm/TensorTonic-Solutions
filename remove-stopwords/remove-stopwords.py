def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    output = []
    stopwords_set = set(stopwords) # clean set usage

    for tok in tokens:
        if tok not in stopwords_set: # reduce to O(1)
            output.append(tok)

    return output