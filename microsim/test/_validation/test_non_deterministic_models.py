"""Confirm model results do not vary significantly across Person representations."""

import numpy as np
from microsim.cohort_risk_model_repository import CohortRiskModelRepository
from microsim.test._validation.fixture import StorePopulationValidationFixture


class TestNonDeterministicModels(StorePopulationValidationFixture):
    def setUp(self):
        super().setUp()
        self._cohort_risk_model_repository = CohortRiskModelRepository()
        self._random_seed = 60632844
        np.random.seed(self._random_seed)

    def test_cohort_physical_activity_model(self):
        model = self._cohort_risk_model_repository.get_model("anyPhysicalActivity")
        vec_df = self.vec_pop.get_people_current_state_and_summary_as_dataframe()
        cur_pop = self.store_pop.person_store.get_population_at(0)

        for (_, vec_row), store_person in zip(vec_df.iterrows(), cur_pop):
            model_random_state = np.random.get_state()
            vec_model_result = model.estimate_next_risk_vectorized(vec_row)
            np.random.set_state(model_random_state)
            store_model_result = model.estimate_next_risk_vectorized(store_person)

            self.assertEqual(vec_model_result, store_model_result)
