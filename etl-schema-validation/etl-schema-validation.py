def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    results = []
    for i, record in enumerate(records):
        errors = []
        for rule in schema:
            col = rule["column"]
            expected_type_str = rule["type"]
            nullable = rule.get("nullable", False)
            
            if col not in record:
                errors.append(f"{col}: missing")
                continue
            
            val = record[col]
            
            if val is None:
                if not nullable:
                    errors.append(f"{col}: null")
                continue
            
            actual_type = type(val)
            actual_type_str = actual_type.__name__
            
            is_type_valid = False
            if expected_type_str == "float":
                if actual_type in (float, int):
                    is_type_valid = True
            else:
                if actual_type_str == expected_type_str:
                    is_type_valid = True
            
            if not is_type_valid:
                errors.append(f"{col}: expected {expected_type_str}, got {actual_type_str}")
                continue
            
            if expected_type_str in ("int", "float"):
                min_val = rule.get("min")
                max_val = rule.get("max")
                if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
                    errors.append(f"{col}: out of range")
        
        results.append((i, len(errors) == 0, errors))
    return results