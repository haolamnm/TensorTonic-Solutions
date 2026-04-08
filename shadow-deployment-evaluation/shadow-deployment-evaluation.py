import math

def evaluate_shadow(production_log, shadow_log, criteria):
    n = len(production_log)
    if n == 0:
        return {"promote": False, "metrics": {}}
        
    prod_correct = sum(1 for log in production_log if log['prediction'] == log['actual'])
    prod_accuracy = prod_correct / n
    
    shadow_correct = sum(1 for log in shadow_log if log['prediction'] == log['actual'])
    shadow_accuracy = shadow_correct / n
    
    accuracy_gain = shadow_accuracy - prod_accuracy
    
    agreements = sum(1 for p, s in zip(production_log, shadow_log) 
                     if p['prediction'] == s['prediction'])
    agreement_rate = agreements / n
    
    shadow_latencies = sorted([log['latency_ms'] for log in shadow_log])
    p95_index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = shadow_latencies[p95_index]
    
    meets_accuracy = accuracy_gain >= criteria['min_accuracy_gain']
    meets_latency = shadow_latency_p95 <= criteria['max_latency_p95']
    meets_agreement = agreement_rate >= criteria['min_agreement_rate']
    
    promote = bool(meets_accuracy and meets_latency and meets_agreement)

    return {
        "promote": promote, 
        "metrics": { 
            "shadow_accuracy": shadow_accuracy, 
            "production_accuracy": prod_accuracy, 
            "accuracy_gain": accuracy_gain, 
            "shadow_latency_p95": shadow_latency_p95, 
            "agreement_rate": agreement_rate 
        }
    }