def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    output = []

    for tok in tokens:
        if tok not in stopwords:
            output.append(tok)

    return output