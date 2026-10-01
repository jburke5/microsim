from microsim.person.person import Person
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.outcomes.outcome import Outcome
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.outcomes.outcome import OutcomeType
from microsim.risk_factors.education import Education
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.outcomes.cognition_model import GCPModel
from microsim.outcomes.dementia_model import DementiaModel
from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.test.outcome_models_repositories import *
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.population.population_factory import PopulationFactory
from microsim.risk_factors.cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository,
                                                   CohortStaticRiskFactorModelRepository)
from microsim.default_treatments.default_treatment_model_repository import DefaultTreatmentModelRepository

import unittest
import numpy as np
import pandas as pd

class TestPersonWaveStatus(unittest.TestCase):
    def setUp(self):
        xoldJoe = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 60,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 140,
                               DynamicRiskFactorsType.DBP.value: 90,
                               DynamicRiskFactorsType.A1C.value: 5.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 200,
                               DynamicRiskFactorsType.BMI.value: 25,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 45,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "oldJoe"}, index=[0])
        self.oldJoe = PersonFactory.get_nhanes_person(xoldJoe.iloc[0], InitializationModelRepository())
        self.oldJoe._afib = [False]

        xyoungJoe = pd.DataFrame({DynamicRiskFactorsType.AGE.value: 40,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.NON_HISPANIC_BLACK.value,
                               DynamicRiskFactorsType.SBP.value: 140,
                               DynamicRiskFactorsType.DBP.value: 90,
                               DynamicRiskFactorsType.A1C.value: 5.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 200,
                               DynamicRiskFactorsType.BMI.value: 25,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 45,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": "youngJoe"}, index=[0])
        self.youngJoe = PersonFactory.get_nhanes_person(xyoungJoe.iloc[0], InitializationModelRepository())
        self.youngJoe._afib = [False]

    def testStatusAfterFatalStroke(self):
        self.youngJoe.advance(2, CohortDynamicRiskFactorModelRepository(), 
                                 DefaultTreatmentModelRepository(), 
                                   AgeOver50CausesFatalStroke(),
                                   None)
        self.assertEqual(41, self.youngJoe._age[-1])
        self.assertEqual(False, self.youngJoe.is_dead)
        self.assertEqual(False, self.youngJoe._stroke)

        self.youngJoe.advance(1, CohortDynamicRiskFactorModelRepository(),
                                   DefaultTreatmentModelRepository(),
                                   AgeOver50CausesFatalStroke(),
                                   None)

        self.oldJoe.advance(2, CohortDynamicRiskFactorModelRepository(),  
                                   DefaultTreatmentModelRepository(),
                                   AgeOver50CausesFatalStroke(),
                                   None)
        self.assertEqual(60, self.oldJoe._age[-1])
        self.assertEqual(True, self.oldJoe.is_dead)
        self.assertEqual(True, self.oldJoe._stroke)

    def testNonCVMortalityLeadsToCorrectStatus(self):
        self.youngJoe.advance(2, CohortDynamicRiskFactorModelRepository(),  
                                   DefaultTreatmentModelRepository(),
                                   AgeOver50CausesNonCVMortality(),
                                   None)
        self.assertEqual(41, self.youngJoe._age[-1])
        self.assertEqual(False, self.youngJoe.is_dead)
        self.assertEqual(False, self.youngJoe._stroke)

        self.youngJoe.advance(1, CohortDynamicRiskFactorModelRepository(),
                                   DefaultTreatmentModelRepository(),
                                   AgeOver50CausesNonCVMortality(),
                                   None)

        self.oldJoe.advance(2, CohortDynamicRiskFactorModelRepository(),
                                   DefaultTreatmentModelRepository(),
                                   AgeOver50CausesNonCVMortality(),
                                   None)
        self.assertEqual(60, self.oldJoe._age[-1])
        self.assertEqual(True, self.oldJoe.is_dead)
        self.assertEqual(False, self.oldJoe._stroke)

    def testHasFatalStrokeInWaveIsCaptured(self):
        self.oldJoe.advance(2, CohortDynamicRiskFactorModelRepository(),
                                   DefaultTreatmentModelRepository(),
                                   AgeOver50CausesFatalStroke(),
                                   None)
        self.assertEqual(True, self.oldJoe.has_stroke_during_simulation())
        self.assertEqual(True, self.oldJoe.has_stroke_during_wave(0))
        with self.assertRaises(RuntimeError):
            self.oldJoe.has_stroke_during_wave(1)
        #Q: this is problematic...
        # calling to make sure it does NOT raise an exception...you should
        # be able to ask whether somebody has a stroke in a wave after they are dead, the answer is
        # "No"
        #self.assertEqual(False, self.oldJoe.has_stroke_during_wave(2))

    def testNonFatalStrokeInWaveWithNonCVDeathIsCaptured(self):
        self.oldJoe.advance(1, CohortDynamicRiskFactorModelRepository(),
                                   DefaultTreatmentModelRepository(),
                                   NonFatalStrokeAndNonCVMortality(),
                                   None)
        self.assertEqual(True, self.oldJoe.has_stroke_during_simulation())
        self.assertEqual(True, self.oldJoe.has_stroke_during_wave(0))
        self.assertEqual(True, self.oldJoe.is_dead)


if __name__ == "__main__":
    unittest.main()
