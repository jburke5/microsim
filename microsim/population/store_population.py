from microsim.outcome import OutcomeType


def is_alive(person):
    return person.current.alive


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
        bp_treatment_recalibration,
        risk_factor_prop_names,
        treatment_prop_names,
        outcome_prop_names,
    ):
        self._person_store = person_store
        self._risk_model_repository = risk_model_repository
        self._outcome_model_repository = outcome_model_repository
        self._bp_treatment_strategy = bp_treatment_strategy
        self._bp_treatment_recalibration = bp_treatment_recalibration

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
            alive_pop = self._person_store.get_population_at(t).where(is_alive)

            if alive_pop.num_persons == 0:
                break

            for person in alive_pop:
                self._advance_person_risk_factors(person)
                self._advance_person_treatments(person)
                self._advance_person_outcomes(person)

            self._recalibrate_treatment(alive_pop)

        self._current_tick = end_tick

    def _advance_person_risk_factors(self, person):
        for rf in self._risk_factor_prop_names + self._treatment_prop_names:
            rf_model = self._risk_model_repository.get_model(rf)
            next_value = rf_model.estimate_next_risk_vectorized(person)
            setattr(person.next, rf, next_value)

    def _advance_person_treatments(self, person):
        self._bp_treatment_strategy.apply_treatment(person.current, person.next)

    def _advance_person_outcomes(self, person):
        cv_outcome = self._outcome_model_repository.get_cv_outcome_for_person(person)
        if cv_outcome is not None:
            if cv_outcome.type == OutcomeType.MI:
                person.next.mi = cv_outcome
            elif cv_outcome.type == OutcomeType.STROKE:
                person.next.stroke = cv_outcome
            else:
                raise ValueError(f"Unhandled cardiovascular outcome type: {cv_outcome.type}")

        next_gcp = self._outcome_model_repository.get_gcp_vectorized(person)
        person.next.gcp = next_gcp

        dementia_outcome = self._outcome_model_repository.get_dementia_for_person(person)
        if dementia_outcome is not None:
            if dementia_outcome.type == OutcomeType.DEMENTIA:
                person.next.dementia = dementia_outcome
            else:
                raise ValueError(f"Unhandled dementia outcome type: {dementia_outcome.type}")

        self._update_liveness(person)

    def _update_liveness(self, person):
        assert person.next.dementia is None or not person.next.dementia.fatal

        will_have_fatal_cv_event = person.next.mi is not None and person.next.mi.fatal
        will_have_fatal_cv_event |= person.next.stroke is not None and person.next.stroke.fatal
        if will_have_fatal_cv_event:
            person.next.alive = False
            return

        will_have_non_cv_death = self._outcome_model_repository.assign_non_cv_mortality_vectorized(
            person
        )
        person.next.alive = not will_have_non_cv_death

    def _recalibrate_treatment(self, alive_pop):
        def set_scratch_next_to_baseline(pop):
            scratch_pop = pop.with_scratch_next()
            for scratch_person in scratch_pop:
                self._advance_person_risk_factors(scratch_person)
                self._advance_person_outcomes(scratch_person)

        self._bp_treatment_recalibration.recalibrate(alive_pop, set_scratch_next_to_baseline)
