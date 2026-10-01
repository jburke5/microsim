import unittest

from microsim.trials.trial_description import KaiserTrialDescription
from microsim.population.population_factory import PopulationFactory
from microsim.population.population import Population
from microsim.population.population_model_repository import PopulationRepositoryType
from microsim.outcomes.outcome import OutcomeType


class TestKaiserTrialDescriptionArgs(unittest.TestCase):
    """Tests that KaiserTrialDescription correctly separates peopleArgs from modelRepoArgs."""

    def test_people_args_has_n_and_person_filters(self):
        td = KaiserTrialDescription(sampleSize=500, personFilters=None)
        self.assertEqual(td.peopleArgs["n"], 500)
        self.assertIn("personFilters", td.peopleArgs)

    def test_people_args_does_not_have_risk_scaling(self):
        td = KaiserTrialDescription(riskScaling={OutcomeType.CARDIOVASCULAR: 2.0})
        self.assertNotIn("riskScaling", td.peopleArgs)

    def test_people_args_does_not_have_wmh_specific(self):
        td = KaiserTrialDescription(wmhSpecific=False)
        self.assertNotIn("wmhSpecific", td.peopleArgs)

    def test_model_repo_args_has_wmh_specific(self):
        td = KaiserTrialDescription(wmhSpecific=False)
        self.assertEqual(td.modelRepoArgs["wmhSpecific"], False)

    def test_model_repo_args_has_risk_scaling(self):
        scaling = {OutcomeType.CARDIOVASCULAR: 1.5, OutcomeType.DEMENTIA: 0.8}
        td = KaiserTrialDescription(riskScaling=scaling)
        self.assertEqual(td.modelRepoArgs["riskScaling"], scaling)

    def test_model_repo_args_default_risk_scaling_is_none(self):
        td = KaiserTrialDescription()
        self.assertIsNone(td.modelRepoArgs["riskScaling"])

    def test_model_repo_args_default_wmh_specific_is_true(self):
        td = KaiserTrialDescription()
        self.assertTrue(td.modelRepoArgs["wmhSpecific"])


class TestKaiserTrialPopulationRiskScaling(unittest.TestCase):
    """Tests that riskScaling flows through to the actual models in Kaiser trial populations."""

    @classmethod
    def setUpClass(cls):
        cls._people = PopulationFactory.get_kaiser_people(n=10)

    def _get_outcome_model_repo(self, riskScaling=None, wmhSpecific=True):
        modelRepo = PopulationFactory.get_kaiser_population_model_repo(
            wmhSpecific=wmhSpecific, riskScaling=riskScaling
        )
        pop = Population(self._people.copy(), modelRepo)
        return pop._modelRepository[PopulationRepositoryType.OUTCOMES.value]

    def test_cv_scaling_reaches_male_model(self):
        repo = self._get_outcome_model_repo(riskScaling={OutcomeType.CARDIOVASCULAR: 2.0})
        maleModel = repo._repository[OutcomeType.CARDIOVASCULAR]._models["male"]
        self.assertEqual(maleModel._riskScaling, 2.0)

    def test_cv_scaling_reaches_female_model(self):
        repo = self._get_outcome_model_repo(riskScaling={OutcomeType.CARDIOVASCULAR: 2.0})
        femaleModel = repo._repository[OutcomeType.CARDIOVASCULAR]._models["female"]
        self.assertEqual(femaleModel._riskScaling, 2.0)

    def test_dementia_scaling_reaches_nhanes_model(self):
        repo = self._get_outcome_model_repo(riskScaling={OutcomeType.DEMENTIA: 0.7})
        nhanesModel = repo._repository[OutcomeType.DEMENTIA]._models["NHANES"]
        self.assertEqual(nhanesModel._riskScaling, 0.7)

    def test_dementia_scaling_reaches_brain_scan_model(self):
        repo = self._get_outcome_model_repo(riskScaling={OutcomeType.DEMENTIA: 0.7})
        brainScanModel = repo._repository[OutcomeType.DEMENTIA]._models["brainScan"]
        self.assertEqual(brainScanModel._riskScaling, 0.7)

    def test_no_scaling_defaults_to_one(self):
        repo = self._get_outcome_model_repo()
        cvModel = repo._repository[OutcomeType.CARDIOVASCULAR]._models["male"]
        dementiaModel = repo._repository[OutcomeType.DEMENTIA]._models["NHANES"]
        self.assertEqual(cvModel._riskScaling, 1.0)
        self.assertEqual(dementiaModel._riskScaling, 1.0)

    def test_cv_scaling_does_not_affect_dementia(self):
        repo = self._get_outcome_model_repo(riskScaling={OutcomeType.CARDIOVASCULAR: 3.0})
        dementiaModel = repo._repository[OutcomeType.DEMENTIA]._models["NHANES"]
        self.assertEqual(dementiaModel._riskScaling, 1.0)

    def test_dementia_scaling_does_not_affect_cv(self):
        repo = self._get_outcome_model_repo(riskScaling={OutcomeType.DEMENTIA: 3.0})
        cvModel = repo._repository[OutcomeType.CARDIOVASCULAR]._models["male"]
        self.assertEqual(cvModel._riskScaling, 1.0)

    def test_cv_risk_value_is_scaled(self):
        baselineRepo = self._get_outcome_model_repo()
        scaledRepo = self._get_outcome_model_repo(riskScaling={OutcomeType.CARDIOVASCULAR: 1.5})
        person = self._people.iloc[0]
        baselineModel = baselineRepo._repository[
            OutcomeType.CARDIOVASCULAR
        ].select_outcome_model_for_person(person)
        scaledModel = scaledRepo._repository[
            OutcomeType.CARDIOVASCULAR
        ].select_outcome_model_for_person(person)
        baselineRisk = baselineModel.get_risk_for_person(person)
        scaledRisk = scaledModel.get_risk_for_person(person)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 1.5, places=10)

    def test_both_scalings_applied_together(self):
        repo = self._get_outcome_model_repo(
            riskScaling={OutcomeType.CARDIOVASCULAR: 1.5, OutcomeType.DEMENTIA: 0.8}
        )
        cvModel = repo._repository[OutcomeType.CARDIOVASCULAR]._models["male"]
        dementiaModel = repo._repository[OutcomeType.DEMENTIA]._models["NHANES"]
        self.assertEqual(cvModel._riskScaling, 1.5)
        self.assertEqual(dementiaModel._riskScaling, 0.8)


class TestKaiserPeopleArgsSampleSize(unittest.TestCase):
    """Tests that peopleArgs correctly controls person creation."""

    def test_sample_size_controls_population_size(self):
        people_small = PopulationFactory.get_kaiser_people(n=20)
        people_large = PopulationFactory.get_kaiser_people(n=50)
        self.assertEqual(people_small.shape[0], 20)
        self.assertEqual(people_large.shape[0], 50)


if __name__ == "__main__":
    unittest.main()
