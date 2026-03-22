def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    results = []

    for req in requests:
        user_id = req["user_id"]
        online_feat = req["online_features"]

        offline_feat = feature_store.get(user_id, defaults)
        combined = {**offline_feat, **online_feat}
        results.append(combined)

    return results
        