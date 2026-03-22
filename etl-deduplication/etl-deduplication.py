def pick_first(records):
    return records[0]

def pick_last(records):
    return records[-1]

def pick_most_complete(records):
    return min(records, key=lambda r: sum(v is None for v in r.values()))

STRATEGIES = {
    "first":         pick_first,
    "last":          pick_last,
    "most_complete": pick_most_complete,
}

def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    groups = {}
    for record in records:
        k = tuple(record[key] for key in key_columns)
        groups.setdefault(k, []).append(record)

    picker = STRATEGIES[strategy]
    return [picker(group) for group in groups.values()]