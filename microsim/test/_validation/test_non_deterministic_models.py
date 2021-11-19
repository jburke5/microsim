"""Confirm model results do not vary significantly across Person representations."""

from itertools import chain
import secrets
import numpy as np
from microsim.outcome import Outcome, OutcomeType
from microsim.population import StorePopulation
from microsim.test._validation.fixture import StorePopulationValidationFixture


class TestNonDeterministicModels(StorePopulationValidationFixture):
    def setUp(self):
        TestNonDeterministicModels._loader_seed = 338275772
        self.initial_person_records = TestNonDeterministicModels.get_or_init_person_records()
        super().setUp()
        self._random_seed = 60632844
        np.random.seed(self._random_seed)

    def test_cohort_afib_model(self):
        model = self.risk_model_repository.get_model("afib")
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        cur_pop = self.store_pop.person_store.get_population_at(0)

        for (_, vec_row), store_person in zip(vec_df.iterrows(), cur_pop):
            model_random_state = np.random.get_state()
            vec_model_result = model.estimate_next_risk_vectorized(vec_row)
            np.random.set_state(model_random_state)
            store_model_result = model.estimate_next_risk_vectorized(store_person)

            self.assertEqual(vec_model_result, store_model_result)

    def test_cohort_antihypertensive_count_model(self):
        model = self.risk_model_repository.get_model("antiHypertensiveCount")
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        cur_pop = self.store_pop.person_store.get_population_at(0)

        for (_, vec_row), store_person in zip(vec_df.iterrows(), cur_pop):
            model_random_state = np.random.get_state()
            vec_model_result = model.estimate_next_risk_vectorized(vec_row)
            np.random.set_state(model_random_state)
            store_model_result = model.estimate_next_risk_vectorized(store_person)

            self.assertEqual(vec_model_result, store_model_result)

    def test_cohort_physical_activity_model(self):
        model = self.risk_model_repository.get_model("anyPhysicalActivity")
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        cur_pop = self.store_pop.person_store.get_population_at(0)

        for (_, vec_row), store_person in zip(vec_df.iterrows(), cur_pop):
            model_random_state = np.random.get_state()
            vec_model_result = model.estimate_next_risk_vectorized(vec_row)
            np.random.set_state(model_random_state)
            store_model_result = model.estimate_next_risk_vectorized(store_person)

            self.assertEqual(vec_model_result, store_model_result)

    def test_cohort_statin_model(self):
        model = self.risk_model_repository.get_model("statin")
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        cur_pop = self.store_pop.person_store.get_population_at(0)

        for (_, vec_row), store_person in zip(vec_df.iterrows(), cur_pop):
            model_random_state = np.random.get_state()
            vec_model_result = model.estimate_next_risk_vectorized(vec_row)
            np.random.set_state(model_random_state)
            store_model_result = model.estimate_next_risk_vectorized(store_person)

            self.assertEqual(vec_model_result, store_model_result)

    def test_cvd_event_model(self):
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        cur_pop = self.store_pop.person_store.get_population_at(0)

        for (_, vec_row), store_person in zip(vec_df.iterrows(), cur_pop):
            model_random_state = np.random.get_state()
            vec_outcome = vec_row_to_outcome_obj(
                self.outcome_model_repository.assign_cv_outcome_vectorized(vec_row)
            )
            np.random.set_state(model_random_state)
            store_outcome = self.outcome_model_repository.get_cv_outcome_for_person(store_person)

            self.assertEqual(vec_outcome, store_outcome)

    def test_non_cvd_death_model(self):
        model = self.outcome_model_repository.assign_non_cv_mortality_vectorized
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        cur_pop = self.store_pop.person_store.get_population_at(0)

        for (_, vec_row), store_person in zip(vec_df.iterrows(), cur_pop):
            model_random_state = np.random.get_state()
            vec_model_result = model(vec_row)
            np.random.set_state(model_random_state)
            store_model_result = model(store_person)

            self.assertEqual(vec_model_result, store_model_result)

    def test_recalibration(self):
        no_recal_store_pop = StorePopulation(
            self.store_pop.person_store,
            self.risk_model_repository,
            self.outcome_model_repository,
            self.bp_treatment_strategy,
            None,  # None = disable BP Treatment recalibration
            self.qaly_assignment_strategy,
            self.store_pop._risk_factor_prop_names,
            self.store_pop._treatment_prop_names,
            self.store_pop._outcome_prop_names,
        )
        no_recal_store_pop.advance()
        cur_pop = no_recal_store_pop.person_store.get_population_at(0).where(lambda p: p.alive)

        # copy advanced values to vectorized population dataframe
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        for n in chain(
            self.store_pop._risk_factor_prop_names,
            self.store_pop._treatment_prop_names,
            ["bpMedsAdded", "gcp"],
        ):
            vec_df[f"{n}Next"] = [getattr(p.next, n) for p in cur_pop]

        vec_df["miNext"] = [p.next.mi is not None for p in cur_pop]
        vec_df["miFatal"] = [p.next.mi is not None and p.next.mi.fatal for p in cur_pop]
        vec_df["ageAtFirstMI"] = [p.ageAtFirstMI for p in cur_pop]
        vec_df["strokeNext"] = [p.next.stroke is not None for p in cur_pop]
        vec_df["strokeFatal"] = [
            p.next.stroke is not None and p.next.stroke.fatal for p in cur_pop
        ]
        vec_df["ageAtFirstStroke"] = [p.ageAtFirstStroke for p in cur_pop]
        vec_df["deadNext"] = vec_df["miFatal"] | vec_df["strokeFatal"]
        vec_df["dementiaNext"] = [p.next.dementia is not None for p in cur_pop]
        vec_df["dementia"] = [p.next.dementia is not None or p.dementia for p in cur_pop]
        vec_df["ageAtFirstDementia"] = [p.ageAtFirstDementia for p in cur_pop]

        # recalibrate vectorized population
        model_random_state = np.random.get_state()
        recalibrated_vec_df = self.vec_pop.apply_recalibration_standards(vec_df)

        # recalibrate store population
        def set_scratch_next_to_baseline(pop):
            scratch_pop = pop.with_scratch_next()
            for scratch_person in scratch_pop:
                no_recal_store_pop._advance_person_risk_factors(scratch_person)
            # reset RNG here to avoid non-det. risk factor models' advancing the RNG
            np.random.set_state(model_random_state)

        self.bp_treatment_recalibration.recalibrate(cur_pop, set_scratch_next_to_baseline)
        recalibrated_store_pop = cur_pop

        for (_, vec_row), store_person in zip(
            recalibrated_vec_df.iterrows(), recalibrated_store_pop
        ):
            self.assertEqual(vec_row.miNext, store_person.next.mi is not None)
            self.assertEqual(
                vec_row.miFatal, store_person.next.mi is not None and store_person.next.mi.fatal
            )
            self.assertEqual(vec_row.strokeNext, store_person.next.stroke is not None)
            self.assertEqual(
                vec_row.strokeFatal,
                store_person.next.stroke is not None and store_person.next.stroke.fatal,
            )


def vec_row_to_outcome_obj(vec_row):
    assert not (vec_row.miNext and vec_row.strokeNext)

    if vec_row.miNext:
        return Outcome(OutcomeType.MI, vec_row.miFatal)
    elif vec_row.strokeNext:
        return Outcome(OutcomeType.STROKE, vec_row.miFatal)
    else:
        return None
