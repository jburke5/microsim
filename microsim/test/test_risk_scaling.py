import unittest
import pandas as pd

from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.outcomes.cv_model import CVModelMale, CVModelFemale
from microsim.outcomes.cv_model_repository import CVModelRepository
from microsim.outcomes.dementia_model import DementiaModel
from microsim.outcomes.dementia_model_repository import DementiaModelRepository
from microsim.outcomes.cognition_outcome import CognitionOutcome
from microsim.outcomes.wmh_outcome import WMHOutcome
from microsim.outcomes.wmh_severity import WMHSeverity


class TestRiskScaling(unittest.TestCase):
    def setUp(self):
        self.x_male = pd.DataFrame(
            {
                DynamicRiskFactorsType.AGE.value: 55,
                StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
                DynamicRiskFactorsType.SBP.value: 120,
                DynamicRiskFactorsType.DBP.value: 80,
                DynamicRiskFactorsType.A1C.value: 6,
                DynamicRiskFactorsType.HDL.value: 50,
                DynamicRiskFactorsType.TOT_CHOL.value: 213,
                DynamicRiskFactorsType.BMI.value: 26.6,
                DynamicRiskFactorsType.LDL.value: 90,
                DynamicRiskFactorsType.TRIG.value: 150,
                DynamicRiskFactorsType.WAIST.value: 34,
                DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                DefaultTreatmentsType.STATIN.value: 0,
                DynamicRiskFactorsType.CREATININE.value: 0,
                "name": "white_male",
            },
            index=[0],
        )
        self._male = PersonFactory.get_nhanes_person(
            self.x_male.iloc[0], InitializationModelRepository()
        )
        self._male._afib = [False]

        self.x_female = pd.DataFrame(
            {
                DynamicRiskFactorsType.AGE.value: 55,
                StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
                StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
                DynamicRiskFactorsType.SBP.value: 120,
                DynamicRiskFactorsType.DBP.value: 80,
                DynamicRiskFactorsType.A1C.value: 6,
                DynamicRiskFactorsType.HDL.value: 50,
                DynamicRiskFactorsType.TOT_CHOL.value: 213,
                DynamicRiskFactorsType.BMI.value: 26.6,
                DynamicRiskFactorsType.LDL.value: 90,
                DynamicRiskFactorsType.TRIG.value: 150,
                DynamicRiskFactorsType.WAIST.value: 34,
                DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                DefaultTreatmentsType.STATIN.value: 0,
                DynamicRiskFactorsType.CREATININE.value: 0,
                "name": "white_female",
            },
            index=[0],
        )
        self._female = PersonFactory.get_nhanes_person(
            self.x_female.iloc[0], InitializationModelRepository()
        )
        self._female._afib = [False]

        # dementia model needs cognition and WMH outcomes
        self.x_dementia_person = pd.DataFrame(
            {
                DynamicRiskFactorsType.AGE.value: 65,
                StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
                DynamicRiskFactorsType.SBP.value: 120,
                DynamicRiskFactorsType.DBP.value: 80,
                DynamicRiskFactorsType.A1C.value: 6,
                DynamicRiskFactorsType.HDL.value: 50,
                DynamicRiskFactorsType.TOT_CHOL.value: 213,
                DynamicRiskFactorsType.BMI.value: 26.6,
                DynamicRiskFactorsType.LDL.value: 90,
                DynamicRiskFactorsType.TRIG.value: 150,
                DynamicRiskFactorsType.WAIST.value: 94,
                DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
                StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                DefaultTreatmentsType.STATIN.value: 0,
                DynamicRiskFactorsType.CREATININE.value: 0,
                "name": "dementia_test_person",
            },
            index=[0],
        )
        self._dementia_person = PersonFactory.get_nhanes_person(
            self.x_dementia_person.iloc[0], InitializationModelRepository()
        )
        self._dementia_person._afib = [False]
        self._dementia_person._outcomes[OutcomeType.COGNITION] = []
        self._dementia_person.add_outcome(CognitionOutcome(False, False, 50.0))
        self._dementia_person.add_outcome(CognitionOutcome(False, False, 49.0))
        self._dementia_person.add_outcome(
            WMHOutcome(
                False, sbi=False, wmh=False, wmhSeverityUnknown=False, wmhSeverity=WMHSeverity.NO
            )
        )

    # --- CV Model direct tests ---

    def test_cv_model_default_scaling_is_one(self):
        model = CVModelMale()
        self.assertEqual(model._riskScaling, 1.0)

    def test_cv_model_stores_scaling(self):
        model = CVModelMale(riskScaling=2.0)
        self.assertEqual(model._riskScaling, 2.0)

    def test_cv_male_scaling_doubles_risk(self):
        baseline = CVModelMale()
        scaled = CVModelMale(riskScaling=2.0)
        baselineRisk = baseline.get_risk_for_person(self._male)
        scaledRisk = scaled.get_risk_for_person(self._male)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 2.0, places=10)

    def test_cv_female_scaling_doubles_risk(self):
        baseline = CVModelFemale()
        scaled = CVModelFemale(riskScaling=2.0)
        baselineRisk = baseline.get_risk_for_person(self._female)
        scaledRisk = scaled.get_risk_for_person(self._female)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 2.0, places=10)

    def test_cv_model_scaling_halves_risk(self):
        baseline = CVModelMale()
        scaled = CVModelMale(riskScaling=0.5)
        baselineRisk = baseline.get_risk_for_person(self._male)
        scaledRisk = scaled.get_risk_for_person(self._male)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 0.5, places=10)

    def test_cv_model_scaling_of_one_matches_baseline(self):
        baseline = CVModelMale()
        scaled = CVModelMale(riskScaling=1.0)
        self.assertEqual(
            baseline.get_risk_for_person(self._male),
            scaled.get_risk_for_person(self._male),
        )

    # --- CV Repository tests ---

    def test_cv_repository_passes_scaling_to_models(self):
        repo = CVModelRepository(riskScaling=1.5)
        maleModel = repo.select_outcome_model_for_person(self._male)
        femaleModel = repo.select_outcome_model_for_person(self._female)
        self.assertEqual(maleModel._riskScaling, 1.5)
        self.assertEqual(femaleModel._riskScaling, 1.5)

    def test_cv_repository_default_scaling(self):
        repo = CVModelRepository()
        model = repo.select_outcome_model_for_person(self._male)
        self.assertEqual(model._riskScaling, 1.0)

    # --- Dementia Model direct tests ---

    def test_dementia_model_default_scaling_is_one(self):
        model = DementiaModel()
        self.assertEqual(model._riskScaling, 1.0)

    def test_dementia_model_stores_scaling(self):
        model = DementiaModel(riskScaling=1.5)
        self.assertEqual(model._riskScaling, 1.5)

    def test_dementia_scaling_doubles_risk(self):
        baseline = DementiaModel()
        scaled = DementiaModel(riskScaling=2.0)
        baselineRisk = baseline.get_risk_for_person(self._dementia_person)
        scaledRisk = scaled.get_risk_for_person(self._dementia_person)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 2.0, places=10)

    def test_dementia_scaling_halves_risk(self):
        baseline = DementiaModel()
        scaled = DementiaModel(riskScaling=0.5)
        baselineRisk = baseline.get_risk_for_person(self._dementia_person)
        scaledRisk = scaled.get_risk_for_person(self._dementia_person)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 0.5, places=10)

    def test_dementia_scaling_of_one_matches_baseline(self):
        baseline = DementiaModel()
        scaled = DementiaModel(riskScaling=1.0)
        self.assertEqual(
            baseline.get_risk_for_person(self._dementia_person),
            scaled.get_risk_for_person(self._dementia_person),
        )

    # --- Dementia Repository tests ---

    def test_dementia_repository_passes_scaling_to_models(self):
        repo = DementiaModelRepository(riskScaling=1.5)
        model = repo.select_outcome_model_for_person(self._dementia_person)
        self.assertEqual(model._riskScaling, 1.5)

    def test_dementia_repository_default_scaling(self):
        repo = DementiaModelRepository()
        model = repo.select_outcome_model_for_person(self._dementia_person)
        self.assertEqual(model._riskScaling, 1.0)

    # --- OutcomeModelRepository tests ---

    def test_outcome_model_repository_passes_cv_scaling(self):
        repo = OutcomeModelRepository(riskScaling={OutcomeType.CARDIOVASCULAR: 2.0})
        cvModel = repo._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(
            self._male
        )
        self.assertEqual(cvModel._riskScaling, 2.0)

    def test_outcome_model_repository_passes_dementia_scaling(self):
        repo = OutcomeModelRepository(riskScaling={OutcomeType.DEMENTIA: 0.7})
        dementiaModel = repo._repository[OutcomeType.DEMENTIA].select_outcome_model_for_person(
            self._dementia_person
        )
        self.assertEqual(dementiaModel._riskScaling, 0.7)

    def test_outcome_model_repository_independent_scaling(self):
        """CV scaling should not affect dementia and vice versa."""
        repo = OutcomeModelRepository(riskScaling={OutcomeType.CARDIOVASCULAR: 3.0})
        dementiaModel = repo._repository[OutcomeType.DEMENTIA].select_outcome_model_for_person(
            self._dementia_person
        )
        self.assertEqual(dementiaModel._riskScaling, 1.0)

    def test_outcome_model_repository_both_scalings(self):
        repo = OutcomeModelRepository(
            riskScaling={OutcomeType.CARDIOVASCULAR: 1.5, OutcomeType.DEMENTIA: 0.8}
        )
        cvModel = repo._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(
            self._male
        )
        dementiaModel = repo._repository[OutcomeType.DEMENTIA].select_outcome_model_for_person(
            self._dementia_person
        )
        self.assertEqual(cvModel._riskScaling, 1.5)
        self.assertEqual(dementiaModel._riskScaling, 0.8)

    def test_outcome_model_repository_none_scaling_defaults_to_one(self):
        repo = OutcomeModelRepository(riskScaling=None)
        cvModel = repo._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(
            self._male
        )
        dementiaModel = repo._repository[OutcomeType.DEMENTIA].select_outcome_model_for_person(
            self._dementia_person
        )
        self.assertEqual(cvModel._riskScaling, 1.0)
        self.assertEqual(dementiaModel._riskScaling, 1.0)

    def test_outcome_model_repository_empty_dict_defaults_to_one(self):
        repo = OutcomeModelRepository(riskScaling={})
        cvModel = repo._repository[OutcomeType.CARDIOVASCULAR].select_outcome_model_for_person(
            self._male
        )
        self.assertEqual(cvModel._riskScaling, 1.0)

    # --- End-to-end risk value tests through OutcomeModelRepository ---

    def test_cv_risk_scaled_through_repository(self):
        baselineRepo = OutcomeModelRepository()
        scaledRepo = OutcomeModelRepository(riskScaling={OutcomeType.CARDIOVASCULAR: 1.5})
        baselineModel = baselineRepo._repository[
            OutcomeType.CARDIOVASCULAR
        ].select_outcome_model_for_person(self._male)
        scaledModel = scaledRepo._repository[
            OutcomeType.CARDIOVASCULAR
        ].select_outcome_model_for_person(self._male)
        baselineRisk = baselineModel.get_risk_for_person(self._male)
        scaledRisk = scaledModel.get_risk_for_person(self._male)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 1.5, places=10)

    def test_dementia_risk_scaled_through_repository(self):
        baselineRepo = OutcomeModelRepository()
        scaledRepo = OutcomeModelRepository(riskScaling={OutcomeType.DEMENTIA: 1.5})
        baselineModel = baselineRepo._repository[
            OutcomeType.DEMENTIA
        ].select_outcome_model_for_person(self._dementia_person)
        scaledModel = scaledRepo._repository[OutcomeType.DEMENTIA].select_outcome_model_for_person(
            self._dementia_person
        )
        baselineRisk = baselineModel.get_risk_for_person(self._dementia_person)
        scaledRisk = scaledModel.get_risk_for_person(self._dementia_person)
        self.assertAlmostEqual(scaledRisk, baselineRisk * 1.5, places=10)


if __name__ == "__main__":
    unittest.main()
