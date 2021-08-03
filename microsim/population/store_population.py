def is_alive(person_record):
    return person_record.alive


class StorePopulation:
    """
    Population that uses a PersonStore to store its people.

    Primarily intended for storing Person data in Numpy ndarrays for memory
    efficiency while still having Person-like objects for advancing, updating,
    and analyzing.
    """

    def __init__(
        self,
        person_store,
        risk_model_repository,
        bp_treatment_strategy,
        risk_factor_prop_names,
        treatment_prop_names,
        outcome_prop_names,
    ):
        self._person_store = person_store
        self._risk_model_repository = risk_model_repository
        self._bp_treatment_strategy = bp_treatment_strategy

        self._risk_factor_prop_names = risk_factor_prop_names
        self._treatment_prop_names = treatment_prop_names
        self._outcome_prop_names = outcome_prop_names

        self._current_tick = 0

    @property
    def person_store(self):
        return self._person_store

    @property
    def current_tick(self):
        return self._current_tick

    def advance(self, num_ticks=1):
        """Advance population by a given number of ticks (default: 1)."""
        start_tick = self._current_tick + 1
        end_tick = start_tick + num_ticks
        for t in range(start_tick, end_tick + 1):
            cur_pop, next_pop = self._person_store.get_population_advance_record_window(
                t, condition=is_alive
            )

            if cur_pop.num_persons == 0:
                break

            for cur_record, next_record in zip(cur_pop, next_pop):
                self._advance_person_risk_factors(cur_record, next_record)
                self._advance_person_treatments(cur_record, next_record)

        self._current_tick = end_tick

    def _advance_person_risk_factors(self, cur_record, next_record):
        for rf in self._risk_factor_prop_names:
            rf_model = self._risk_model_repository.get_model(rf)
            next_value = rf_model.get_estimate_next_risk_vectorized(cur_record)
            setattr(next_record, rf, next_value)

    def _advance_person_treatments(self, cur_record, next_record):
        for treatment in self._treatment_prop_names:
            treatment_model = self._risk_model_repository.get_model(treatment)
            next_value = treatment_model.get_estimate_next_risk_vectorized(cur_record)
            setattr(next_record, treatment, next_value)
        self._bp_treatment_strategy.apply_treatment(cur_record, next_record)
