def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    if not models:
        return ""

    # elegant use of max
    best = max(models, key=lambda m: (m["accuracy"], -m["latency"], m["timestamp"]))
    return best["name"]