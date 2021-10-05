import numpy as np
from microsim.outcome import OutcomeType


def is_alive(person_record):
    return person_record.current.alive


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
            alive_pop = self._person_store.get_population_at(t).where(is_alive)

            if alive_pop.num_persons == 0:
                break

            for person in alive_pop:
                self._advance_person_risk_factors(person)
                self._advance_person_treatments(person)
                self._advance_person_outcomes(person)

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
        scratch_pop = alive_pop.get_scratch_copy()
        for scratch_person in scratch_pop:
            self._advance_person_risk_factors(scratch_person)
            self._advance_person_outcomes(scratch_person)

        max_bp_meds = 5
        indices_by_bp_meds = [[] for i in range(max_bp_meds + 1)]
        for i, person in enumerate(alive_pop):
            bp_meds_added = person.next.bpMedsAdded
            indices_by_bp_meds[bp_meds_added].append(i)

        bp_treatment_standards = (
            self._bp_treatment_strategy.get_treatment_recalibration_for_population()
        )
        for num_bp_meds in range(1, max_bp_meds + 1):
            indices = indices_by_bp_meds[num_bp_meds]

            model_relrisk, treated_has_event, untreated_cv_risks = self._get_model_cv_event_stats(
                alive_pop, scratch_pop, indices
            )
            num_mi_events = len(treated_has_event[OutcomeType.MI][True])
            num_stroke_events = len(treated_has_event[OutcomeType.STROKE][True])
            standard_mi_relrisk = bp_treatment_standards[OutcomeType.MI] ** num_bp_meds
            standard_stroke_relrisk = bp_treatment_standards[OutcomeType.STROKE] ** num_bp_meds
            delta_mi_relrisk = model_relrisk[OutcomeType.MI] - standard_mi_relrisk
            delta_stroke_relrisk = model_relrisk[OutcomeType.STROKE] - standard_stroke_relrisk
            num_mis_to_change = int(
                round(delta_mi_relrisk * num_mi_events) / model_relrisk[OutcomeType.MI]
            )
            num_strokes_to_change = int(
                round(delta_stroke_relrisk * num_stroke_events) / model_relrisk[OutcomeType.STROKE]
            )

            if delta_mi_relrisk < 0 and num_mis_to_change > 0:
                no_mi_indices = treated_has_event[OutcomeType.MI][False]
                no_mi_risk = [untreated_cv_risks[i][OutcomeType.MI] for i in no_mi_indices]
                add_mi_indices = np.random.choice(
                    no_mi_indices, size=num_mis_to_change, replace=False, p=no_mi_risk
                )
                for i in add_mi_indices:
                    person = alive_pop[i]
                    person.next.mi = self._outcome_model_repository.new_mi_for_person(person)
            elif delta_mi_relrisk > 0 and num_mis_to_change > 0 and num_mi_events > 0:
                num_mis_to_change = min(num_mis_to_change, num_mi_events)
                has_mi_indices = treated_has_event[OutcomeType.MI][True]
                has_mi_risk = [untreated_cv_risks[i][OutcomeType.MI] for i in has_mi_indices]
                remove_mi_indices = np.random.choice(
                    has_mi_indices, size=num_mis_to_change, replace=False, p=has_mi_risk
                )
                for i in remove_mi_indices:
                    person = alive_pop[i]
                    person.next.mi = None

            if delta_stroke_relrisk < 0 and num_strokes_to_change > 0:
                no_stroke_indices = treated_has_event[OutcomeType.STROKE][False]
                no_stroke_risk = [
                    untreated_cv_risks[i][OutcomeType.STROKE] for i in no_stroke_indices
                ]
                add_stroke_indices = np.random.choice(
                    no_stroke_indices, size=num_strokes_to_change, replace=False, p=no_stroke_risk
                )
                for i in add_stroke_indices:
                    person = alive_pop[i]
                    person.next.stroke = self._outcome_model_repository.new_stroke_for_person(
                        person
                    )
            elif delta_stroke_relrisk > 0 and num_strokes_to_change > 0 and num_stroke_events > 0:
                num_strokes_to_change = min(num_strokes_to_change, num_stroke_events)
                has_stroke_indices = treated_has_event[OutcomeType.STROKE][True]
                has_stroke_risk = [
                    untreated_cv_risks[i][OutcomeType.STROKE] for i in has_stroke_indices
                ]
                remove_stroke_indices = np.random.choice(
                    has_stroke_indices,
                    size=num_strokes_to_change,
                    replace=False,
                    p=has_stroke_risk,
                )
                for i in remove_stroke_indices:
                    person = alive_pop[i]
                    person.next.stroke = None

    def _get_model_cv_event_stats(self, treated_pop, untreated_pop, indices):
        treated_total_mi_risk = 0
        untreated_total_mi_risk = 0
        treated_total_stroke_risk = 0
        untreated_total_stroke_risk = 0
        treated_has_event = {
            OutcomeType.MI: {False: [], True: []},
            OutcomeType.STROKE: {False: [], True: []},
        }
        untreated_cv_risks = []
        for i in indices:
            treated_person = treated_pop[i]
            treated_cv_risks = self._outcome_model_repository.get_cv_event_risks_for_person(
                treated_person
            )
            treated_total_mi_risk += treated_cv_risks[OutcomeType.MI]
            treated_total_stroke_risk += treated_cv_risks[OutcomeType.STROKE]
            will_have_mi = treated_person.next.mi is not None
            treated_has_event[OutcomeType.MI][will_have_mi].append(i)
            will_have_stroke = treated_person.next.stroke is not None
            treated_has_event[OutcomeType.STROKE][will_have_stroke].append(i)

            untreated_person = untreated_pop[i]
            untreated_cv_risks = self._outcome_model_repository.get_cv_event_risks_for_person(
                untreated_person
            )
            untreated_total_mi_risk += untreated_cv_risks[OutcomeType.MI]
            untreated_total_stroke_risk += untreated_cv_risks[OutcomeType.STROKE]
            untreated_cv_risks.append(untreated_cv_risks)

        num_persons = len(indices)
        treated_mean_mi_risk = treated_total_mi_risk / num_persons
        treated_mean_stroke_risk = treated_total_stroke_risk / num_persons
        untreated_mean_mi_risk = untreated_total_mi_risk / num_persons
        untreated_mean_stroke_risk = untreated_total_stroke_risk / num_persons

        model_mi_relrisk = treated_mean_mi_risk / untreated_mean_mi_risk
        model_stroke_relrisk = treated_mean_stroke_risk / untreated_mean_stroke_risk
        model_relrisk = {
            OutcomeType.MI: model_mi_relrisk,
            OutcomeType.STROKE: model_stroke_relrisk,
        }
        return model_relrisk, treated_has_event, untreated_cv_risks
