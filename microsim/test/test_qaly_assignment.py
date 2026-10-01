import unittest
import numpy as np
import pandas as pd

from microsim.person.person import Person
#from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.risk_factors.education import Education
from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.outcomes.outcome_model_repository import OutcomeModelRepository
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.dementia_model import DementiaModel
from microsim.outcomes.cognition_model import GCPModel, GCPStrokeModel
from microsim.population.initialization_repository import InitializationRepository
from microsim.population.population_factory import PopulationFactory
from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.test.static_risk_factor_over_time_repository import StaticDefaultTreatmentModelRepository, StaticRiskFactorOverTimeRepository
from microsim.risk_factors.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.test.outcome_models_repositories import AlwaysNonFatalStroke, AlwaysFatalStroke, AlwaysNonFatalMI, AlwaysDementia, NoOutcome

class TestQALYAssignment(unittest.TestCase):
    def getPerson(self, age=65):

        x = pd.DataFrame({DynamicRiskFactorsType.AGE.value: age,
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
                               StaticRiskFactorsType.RACE_ETHNICITY.value:RaceEthnicity.MEXICAN_AMERICAN.value,
                               DynamicRiskFactorsType.SBP.value: 140,
                               DynamicRiskFactorsType.DBP.value: 80,
                               DynamicRiskFactorsType.A1C.value: 6.5,
                               DynamicRiskFactorsType.HDL.value: 50,
                               DynamicRiskFactorsType.TOT_CHOL.value: 210,
                               DynamicRiskFactorsType.BMI.value: 22,
                               DynamicRiskFactorsType.LDL.value: 90,
                               DynamicRiskFactorsType.TRIG.value: 150,
                               DynamicRiskFactorsType.WAIST.value: 50,
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.FORMER.value,
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
                               DefaultTreatmentsType.STATIN.value: 0,
                               DynamicRiskFactorsType.CREATININE.value: 0,
                               "name": f"person{age}"}, index=[0])
        xPerson = PersonFactory.get_nhanes_person(x.iloc[0], InitializationModelRepository())
        xPerson._afib = [False]
        return xPerson

    def setUp(self):
        self._hasNoConditions = self.getPerson()
        self._hasDementia = self.getPerson()
        self._hasStroke = self.getPerson()
        self._hasFatalStroke = self.getPerson()
        self._hasMI = self.getPerson()
        self._age90 = self.getPerson(90)

    def testAgeQALYsOnly(self):
        self._hasNoConditions.advance(1, None, None, NoOutcome(), None)
        qalys = self._hasNoConditions.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[-1])
 
        self._age90.advance(1, None, None, NoOutcome(), None)
        qalys = self._age90.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(0.8, qalys[-1])

    def testStrokeQALYS(self):
        self._hasStroke.advance(1, None, None, NoOutcome(), None)
        qalys = self._hasStroke.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self._hasStroke.advance(3, StaticRiskFactorOverTimeRepository(), 
                                   StaticDefaultTreatmentModelRepository(), 
                                   AlwaysNonFatalStroke(),
                                   None)
        qalys = self._hasStroke.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self.assertEqual(0.67, qalys[1])
        self.assertEqual(0.9, qalys[2])
        self.assertEqual(0.9, qalys[3])

    def testMIQALYS(self):
        self._hasMI.advance(1, None, None, NoOutcome(), None)
        qalys = self._hasMI.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self._hasMI.advance(3, StaticRiskFactorOverTimeRepository(), 
                                   StaticDefaultTreatmentModelRepository(), 
                                   AlwaysNonFatalMI(),
                                   None)
        qalys = self._hasMI.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self.assertEqual(0.88, qalys[1])
        self.assertEqual(0.9, qalys[2])
        self.assertEqual(0.9, qalys[3])

    def testDementiaQALYS(self):
        self._hasDementia.advance(1, None, None, NoOutcome(), None)
        qalys = self._hasDementia.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self._hasDementia.advance(3, StaticRiskFactorOverTimeRepository(),    
                                   StaticDefaultTreatmentModelRepository(),
                                   AlwaysDementia(),
                                   None)
        qalys = self._hasDementia.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self.assertEqual(0.80, qalys[1])
        self.assertEqual(0.79, qalys[2])
        self.assertEqual(0.78, qalys[3])

    def testQALYSWithMultipleConditions(self):
        self._hasMI.advance(1, None, None, NoOutcome(), None)
        qalys = self._hasMI.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self._hasMI.advance(1, StaticRiskFactorOverTimeRepository(),    
                                   StaticDefaultTreatmentModelRepository(),
                                   AlwaysNonFatalMI(),
                                   None)
        self._hasMI.advance(1, StaticRiskFactorOverTimeRepository(),
                                   StaticDefaultTreatmentModelRepository(),
                                   AlwaysNonFatalStroke(),
                                   None)
        self._hasMI.advance(1, StaticRiskFactorOverTimeRepository(),
                                   StaticDefaultTreatmentModelRepository(),
                                   AlwaysNonFatalMI(),
                                   None)
        qalys = self._hasMI.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self.assertEqual(0.88, qalys[1])
        # 0.9 * 0.678
        self.assertAlmostEqual(0.603, qalys[2], places=5)
        # 0.9 * 0.9
        self.assertEqual(0.81, qalys[3])

    def testQALYsWithDeath(self):
        self._hasFatalStroke.advance(1, None, None, NoOutcome(), None)
        qalys = self._hasFatalStroke.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self._hasFatalStroke.advance(1, StaticRiskFactorOverTimeRepository(), 
                                   StaticDefaultTreatmentModelRepository(), 
                                   AlwaysFatalStroke(),
                                   None)
        qalys = self._hasFatalStroke.get_outcome_item(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")
        self.assertEqual(1, qalys[0])
        self.assertEqual(0, qalys[1])


if __name__ == "__main__":
    unittest.main()
