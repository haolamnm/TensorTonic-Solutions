def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    if not models:
        return ""

    picked = models[0]
    for model in models:
        if model["accuracy"] > picked["accuracy"]:
            picked = model
        elif model["accuracy"] < picked["accuracy"]:
            continue
        else:
            if model["latency"] < picked["latency"]:
                picked = model
            elif model["latency"] > picked["latency"]:
                continue
            else:
                if model["timestamp"] > picked["timestamp"]:
                    picked = model
                else:
                    continue

    return picked["name"]