"""Confirm model results do not vary significantly across Person representations."""

import numpy as np
from microsim.outcome import Outcome, OutcomeType
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


def vec_row_to_outcome_obj(vec_row):
    assert not (vec_row.miNext and vec_row.strokeNext)

    if vec_row.miNext:
        return Outcome(OutcomeType.MI, vec_row.miFatal)
    elif vec_row.strokeNext:
        return Outcome(OutcomeType.STROKE, vec_row.miFatal)
    else:
        return None
