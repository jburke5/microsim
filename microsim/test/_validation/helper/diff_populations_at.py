from microsim.test._validation.helper.person_obj_to_person_record import (
    person_obj_to_person_record,
)
from microsim.test._validation.helper.diff_person_records import diff_person_records


def diff_populations_at(vec_pop, store, t, all_prop_names):
    store_records = (p.current for p in store.get_population_at(t))
    vec_records = (person_obj_to_person_record(p, t) for p in vec_pop._people)

    all_diffs = []
    for i, (vec_record, store_record) in enumerate(zip(vec_records, store_records)):
        diff = diff_person_records(vec_record, store_record, all_prop_names=all_prop_names)
        if diff:
            all_diffs.append((i, diff))
    return all_diffs
