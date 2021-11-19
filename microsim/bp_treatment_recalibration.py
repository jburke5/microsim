import numpy as np
from microsim.outcome import OutcomeType


def new_num_bp_meds_upto_func(max_bp_meds):
    def num_bp_meds_upto(person):
        return min(person.next.bpMedsAdded, max_bp_meds)

    return num_bp_meds_upto


def has_mi(person):
    return person.next.mi is not None


def has_stroke(person):
    return person.next.stroke is not None


class BPTreatmentRecalibration:
    def __init__(self, bp_treatment_strategy, outcome_model_repository):
        self._bp_treatment_standards = (
            bp_treatment_strategy.get_treatment_recalibration_for_population()
        )
        self._outcome_model_repository = outcome_model_repository

    def recalibrate(self, pop, set_scratch_next_to_baseline):
        set_scratch_next_to_baseline(pop)

        max_bp_meds = 5
        num_bp_meds_upto = new_num_bp_meds_upto_func(max_bp_meds)
        all_bp_med_groups = pop.group_by(num_bp_meds_upto)
        recalibration_groups = [
            all_bp_med_groups[i] for i in range(1, max_bp_meds + 1) if i in all_bp_med_groups
        ]

        for num_bp_meds, treated_subpop in enumerate(recalibration_groups, start=1):
            untreated_subpop = treated_subpop.with_scratch_next()
            model_relrisk = self._get_model_cv_event_relrisk(treated_subpop, untreated_subpop)

            standard_mi_relrisk = self._bp_treatment_standards[OutcomeType.MI] ** num_bp_meds
            standard_stroke_relrisk = (
                self._bp_treatment_standards[OutcomeType.STROKE] ** num_bp_meds
            )

            self._recalibrate_stroke(
                treated_subpop, model_relrisk[OutcomeType.STROKE], standard_stroke_relrisk
            )
            self._recalibrate_mi(
                treated_subpop, model_relrisk[OutcomeType.MI], standard_mi_relrisk
            )

    def _recalibrate_mi(self, treated_subpop, model_mi_relrisk, standard_mi_relrisk):
        treated_has_mi = treated_subpop.group_by(has_mi)
        num_mi_events = treated_has_mi[True].num_persons if True in treated_has_mi else 0
        delta_mi_relrisk = model_mi_relrisk - standard_mi_relrisk
        num_mis_to_change = int(round(delta_mi_relrisk * num_mi_events) / model_mi_relrisk)

        if delta_mi_relrisk < 0 and num_mis_to_change > 0:
            self._recalibrate_add_mis(treated_has_mi[False], num_mis_to_change)
        elif delta_mi_relrisk > 0 and num_mis_to_change > 0 and num_mi_events > 0:
            num_mis_to_change = min(num_mis_to_change, num_mi_events)
            self._recalibrate_remove_mis(treated_has_mi[True], num_mis_to_change)

    def _recalibrate_add_mis(self, no_mi_subpop, num_mis_to_change):
        no_mi_untreated_subpop = no_mi_subpop.with_scratch_next()
        no_mi_risks = self._get_model_cv_event_risk(no_mi_untreated_subpop, OutcomeType.MI)
        add_mi_rel_indices = np.random.choice(
            no_mi_subpop.num_persons, size=num_mis_to_change, replace=False, p=no_mi_risks
        )
        for person in (no_mi_subpop[i] for i in add_mi_rel_indices):
            person.next.mi = self._outcome_model_repository.new_mi_for_person(person)

    def _recalibrate_remove_mis(self, has_mi_subpop, num_mis_to_change):
        has_mi_untreated_subpop = has_mi_subpop.with_scratch_next()
        not_have_mi_risk = self._get_model_cv_event_risk(
            has_mi_untreated_subpop, OutcomeType.MI, complement=True
        )
        remove_mi_indices = np.random.choice(
            has_mi_subpop.num_persons, size=num_mis_to_change, replace=False, p=not_have_mi_risk
        )
        for person in (has_mi_subpop[i] for i in remove_mi_indices):
            person.next.mi = None

    def _recalibrate_stroke(self, treated_subpop, model_stroke_relrisk, standard_stroke_relrisk):
        treated_has_stroke = treated_subpop.group_by(has_stroke)
        num_strokes = treated_has_stroke[True].num_persons if True in treated_has_stroke else 0
        delta_stroke_relrisk = model_stroke_relrisk - standard_stroke_relrisk
        num_strokes_to_change = int(
            round(delta_stroke_relrisk * num_strokes) / model_stroke_relrisk
        )

        if delta_stroke_relrisk < 0 and num_strokes_to_change > 0:
            self._recalibrate_add_strokes(treated_has_stroke[False], num_strokes_to_change)
        elif delta_stroke_relrisk > 0 and num_strokes_to_change > 0 and num_strokes > 0:
            num_strokes_to_change = min(num_strokes_to_change, num_strokes)
            self._recalibrate_remove_strokes(treated_has_stroke[True], num_strokes_to_change)

    def _recalibrate_add_strokes(self, no_stroke_subpop, num_strokes_to_change):
        no_stroke_untreated_subpop = no_stroke_subpop.with_scratch_next()
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

    def _recalibrate_remove_strokes(self, has_stroke_subpop, num_strokes_to_change):
        has_stroke_untreated_subpop = has_stroke_subpop.with_scratch_next()
        not_have_stroke_risks = self._get_model_cv_event_risk(
            has_stroke_untreated_subpop, OutcomeType.STROKE, complement=True
        )
        remove_stroke_indices = np.random.choice(
            has_stroke_subpop.num_persons,
            size=num_strokes_to_change,
            replace=False,
            p=not_have_stroke_risks,
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

    def _get_model_cv_event_risk(self, untreated_pop, event_type, complement=False):
        cv_event_risks = np.empty(untreated_pop.num_persons, dtype=np.float64)
        with np.nditer(cv_event_risks, op_flags=["writeonly"]) as it:
            for person, row in zip(untreated_pop, it):
                cv_risks = self._outcome_model_repository.get_cv_event_risks_for_person(person)
                row[...] = cv_risks[event_type]
        if complement:
            cv_event_risks = 1 - cv_event_risks
        normalized_cv_event_risks = cv_event_risks / cv_event_risks.sum()
        return normalized_cv_event_risks
