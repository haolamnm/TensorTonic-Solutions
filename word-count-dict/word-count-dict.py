def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    counter = {}
    
    for sent in sentences:
        for word in sent:
            counter[word] = counter.get(word, 0) + 1

    return dict(counter)