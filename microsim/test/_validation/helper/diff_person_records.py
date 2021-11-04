def diff_person_records(a, b, all_prop_names):
    diff = {}
    for prop_name in all_prop_names:
        a_val = getattr(a, prop_name)
        b_val = getattr(b, prop_name)
        if a_val != b_val:
            diff[prop_name] = (a_val, b_val)
    return diff if diff else None
