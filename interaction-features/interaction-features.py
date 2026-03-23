def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    for x in X:
        interaction = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                interaction.append(x[i] * x[j])
        x.extend(interaction)

    return X