import numpy as np
from microsim.outcome import OutcomeType


def is_alive(person_record):
    return person_record.current.alive


def new_num_bp_meds_upto_func(max_bp_meds):
    def num_bp_meds_upto(person):
        return min(person.next.bpMedsAdded, max_bp_meds)

    return num_bp_meds_upto


def has_mi(person):
    return person.next.mi is not None


def has_stroke(person):
    return person.next.stroke is not None


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
        scratch_pop = alive_pop.get_scratch_copy()
        for scratch_person in scratch_pop:
            self._advance_person_risk_factors(scratch_person)
            self._advance_person_outcomes(scratch_person)

        max_bp_meds = 5
        num_bp_meds_upto = new_num_bp_meds_upto_func(max_bp_meds)
        all_bp_med_groups = alive_pop.group_by(num_bp_meds_upto)
        recalibration_groups = [
            all_bp_med_groups[i] for i in range(1, max_bp_meds + 1) if i in all_bp_med_groups
        ]

        bp_treatment_standards = (
            self._bp_treatment_strategy.get_treatment_recalibration_for_population()
        )
        for num_bp_meds, treated_subpop in enumerate(recalibration_groups, start=1):
            untreated_subpop = treated_subpop.get_scratch_copy()
            model_relrisk = self._get_model_cv_event_relrisk(treated_subpop, untreated_subpop)

            standard_mi_relrisk = bp_treatment_standards[OutcomeType.MI] ** num_bp_meds
            standard_stroke_relrisk = bp_treatment_standards[OutcomeType.STROKE] ** num_bp_meds

            self._recalibrate_mi(
                treated_subpop, model_relrisk[OutcomeType.MI], standard_mi_relrisk
            )
            self._recalibrate_stroke(
                treated_subpop, model_relrisk[OutcomeType.STROKE], standard_stroke_relrisk
            )

    def _recalibrate_mi(self, treated_subpop, model_mi_relrisk, standard_mi_relrisk):
        treated_has_mi = treated_subpop.group_by(has_mi)
        num_mi_events = treated_has_mi[True].num_persons
        delta_mi_relrisk = model_mi_relrisk - standard_mi_relrisk
        num_mis_to_change = int(round(delta_mi_relrisk * num_mi_events) / model_mi_relrisk)

        if delta_mi_relrisk < 0 and num_mis_to_change > 0:
            no_mi_subpop = treated_has_mi[False]
            no_mi_untreated_subpop = no_mi_subpop.get_scratch_copy()
            no_mi_risks = self._get_model_cv_event_risk(no_mi_untreated_subpop, OutcomeType.MI)
            add_mi_rel_indices = np.random.choice(
                no_mi_subpop.num_persons, size=num_mis_to_change, replace=False, p=no_mi_risks
            )
            for person in (no_mi_subpop[i] for i in add_mi_rel_indices):
                person.next.mi = self._outcome_model_repository.new_mi_for_person(person)
        elif delta_mi_relrisk > 0 and num_mis_to_change > 0 and num_mi_events > 0:
            num_mis_to_change = min(num_mis_to_change, num_mi_events)
            has_mi_subpop = treated_has_mi[True]
            has_mi_untreated_subpop = has_mi_subpop.get_scratch_copy()
            has_mi_risk = self._get_model_cv_event_risk(has_mi_untreated_subpop, OutcomeType.MI)
            remove_mi_indices = np.random.choice(
                has_mi_subpop.num_persons, size=num_mis_to_change, replace=False, p=has_mi_risk
            )
            for person in (has_mi_subpop[i] for i in remove_mi_indices):
                person.next.mi = None

    def _recalibrate_stroke(self, treated_subpop, model_stroke_relrisk, standard_stroke_relrisk):
        treated_has_stroke = treated_subpop.group_by(has_stroke)
        num_strokes = treated_has_stroke[True].num_persons
        delta_stroke_relrisk = model_stroke_relrisk - standard_stroke_relrisk
        num_strokes_to_change = int(
            round(delta_stroke_relrisk * num_strokes) / model_stroke_relrisk
        )

        if delta_stroke_relrisk < 0 and num_strokes_to_change > 0:
            no_stroke_subpop = treated_has_stroke[False]
            no_stroke_untreated_subpop = no_stroke_subpop.get_scratch_copy()
            no_stroke_risks = self._get_model_cv_event_risk(
                no_stroke_untreated_subpop, OutcomeType.STROKE
            )
            add_stroke_indices = np.random.choice(
                no_stroke_subpop.num_persons,
                size=num_strokes_to_change,
                replace=False,
                p=no_stroke_risks,
            )
            for person in (no_stroke_subpop[i] for i in add_stroke_indices):
                person.next.stroke = self._outcome_model_repository.new_stroke_for_person(person)
        elif delta_stroke_relrisk > 0 and num_strokes_to_change > 0 and num_strokes > 0:
            num_strokes_to_change = min(num_strokes_to_change, num_strokes)
            has_stroke_subpop = treated_has_stroke[True]
            has_stroke_untreated_subpop = has_stroke_subpop.get_scratch_copy()
            has_stroke_risks = self._get_model_cv_event_risk(
                has_stroke_untreated_subpop, OutcomeType.STROKE
            )
            remove_stroke_indices = np.random.choice(
                has_stroke_subpop.num_persons,
                size=num_strokes_to_change,
                replace=False,
                p=has_stroke_risks,
            )
            for person in (has_stroke_subpop[i] for i in remove_stroke_indices):
                person.next.stroke = None

    def _get_model_cv_event_relrisk(self, treated_pop, untreated_pop):
        treated_total_mi_risk = 0
        untreated_total_mi_risk = 0
        treated_total_stroke_risk = 0
        untreated_total_stroke_risk = 0
        for treated_person, untreated_person in zip(treated_pop, untreated_pop):
            treated_cv_risks = self._outcome_model_repository.get_cv_event_risks_for_person(
                treated_person
            )
            treated_total_mi_risk += treated_cv_risks[OutcomeType.MI]
            treated_total_stroke_risk += treated_cv_risks[OutcomeType.STROKE]

            untreated_cv_risks = self._outcome_model_repository.get_cv_event_risks_for_person(
                untreated_person
            )
            untreated_total_mi_risk += untreated_cv_risks[OutcomeType.MI]
            untreated_total_stroke_risk += untreated_cv_risks[OutcomeType.STROKE]

        model_relrisk = {
            OutcomeType.MI: treated_total_mi_risk / untreated_total_mi_risk,
            OutcomeType.STROKE: treated_total_stroke_risk / untreated_total_stroke_risk,
        }
        return model_relrisk

    def _get_model_cv_event_risk(self, untreated_pop, event_type):
        cv_event_risks = []
        for person in untreated_pop:
            person_cv_risks = self._outcome_model_repository.get_cv_event_risks_for_person(person)
            person_event_risk = person_cv_risks[event_type]
            cv_event_risks.append(person_event_risk)
        total_risk = sum(cv_event_risks)
        if total_risk == 0:
            return cv_event_risks
        normalized_cv_event_risks = [r / total_risk for r in cv_event_risks]
        return normalized_cv_event_risks
