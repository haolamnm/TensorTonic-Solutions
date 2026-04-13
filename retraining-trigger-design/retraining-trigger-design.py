def retraining_policy(daily_stats, config):
    """
    Decide which days to trigger model retraining.
    """
    retrain_days = []
    
    days_since_retrain = 0
    budget = config["budget"]
    has_retrained = False # tracks if the initial cooldown freebie has been used
    
    # extract config values for readability
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    
    for stat in daily_stats:
        day = stat["day"]
        drift_score = stat["drift_score"]
        performance = stat["performance"]
        
        # state carries across days and increments by 1
        days_since_retrain += 1
        
        # check Triggers (ANY condition)
        is_drift = drift_score > drift_threshold
        is_perf = performance < performance_threshold
        is_stale = days_since_retrain >= max_staleness
        
        if is_drift or is_perf or is_stale:
            
            # check constraints (BOTH must be satisfied)
            # cooldown is bypassed if we haven't trained yet
            cooldown_met = (not has_retrained) or (days_since_retrain >= cooldown)
            budget_met = budget >= retrain_cost
            
            # execute retrain & reset state
            if cooldown_met and budget_met:
                retrain_days.append(day)
                budget -= retrain_cost
                days_since_retrain = 0
                has_retrained = True
                
    return retrain_days