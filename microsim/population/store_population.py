from microsim.outcome import OutcomeType


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
        outcome_model_repository,
        bp_treatment_strategy,
        risk_factor_prop_names,
        treatment_prop_names,
        outcome_prop_names,
    ):
        self._person_store = person_store
        self._risk_model_repository = risk_model_repository
        self._outcome_model_repository = outcome_model_repository
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
        start_tick = self._current_tick
        end_tick = start_tick + num_ticks
        for t in range(start_tick, end_tick):
            alive_pop = self._person_store.get_population_at(t, condition=is_alive)

            if alive_pop.num_persons == 0:
                break

            for person in alive_pop:
                self._advance_person_risk_factors(person)
                self._advance_person_treatments(person)
                self._advance_person_outcomes(person)

        self._current_tick = end_tick

    def _advance_person_risk_factors(self, person):
        for rf in self._risk_factor_prop_names:
            rf_model = self._risk_model_repository.get_model(rf)
            next_value = rf_model.estimate_next_risk_vectorized(person)
            setattr(person.next, rf, next_value)

    def _advance_person_treatments(self, person):
        for treatment in self._treatment_prop_names:
            treatment_model = self._risk_model_repository.get_model(treatment)
            next_value = treatment_model.estimate_next_risk_vectorized(person)
            setattr(person.next, treatment, next_value)
        self._bp_treatment_strategy.apply_treatment(person.current, person.next)

    def _advance_person_outcomes(self, person):
        cv_outcome = self._outcome_model_repository.get_cv_outcome_for_person(person)
        if cv_outcome is not None:
            if cv_outcome.type == OutcomeType.MI:
                setattr(person.next, "mi", cv_outcome)
            elif cv_outcome.type == OutcomeType.STROKE:
                setattr(person.next, "stroke", cv_outcome)
            else:
                raise ValueError(f"Unhandled cardiovascular outcome type: {cv_outcome.type}")

        next_alive = not (cv_outcome is not None and cv_outcome.fatal)
        setattr(person.next, "alive", next_alive)

        next_gcp = self._outcome_model_repository.get_gcp_vectorized(person)
        setattr(person.next, "gcp", next_gcp)

        dementia_outcome = self._outcome_model_repository.get_dementia_for_person(person)
        if dementia_outcome is not None:
            if dementia_outcome.type == OutcomeType.DEMENTIA:
                setattr(person.next, "dementia", dementia_outcome)
            else:
                raise ValueError(f"Unhandled dementia outcome type: {dementia_outcome.type}")
