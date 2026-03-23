def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    label2index = {}
    for index, label in enumerate(ordering):
        label2index[label] = index

    encoded = []
    for value in values:
        encoded.append(label2index[value])

    return encoded