import unittest

from microsim.person import Person
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.education import Education
from microsim.outcome import Outcome, OutcomeType
from microsim.test.do_not_change_risk_factors_model_repository import (
    DoNotChangeRiskFactorsModelRepository,
)
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.outcome_model_type import OutcomeModelType
from microsim.dementia_model import DementiaModel
from microsim.gcp_model import GCPModel
from microsim.initialization_repository import InitializationRepository


class AlwaysNonFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()
        self._models = {}
        self._models[OutcomeModelType.DEMENTIA] = DementiaModel()
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, 0)

    def assign_non_cv_mortality(self, person, years=1):
        return False


class AlwaysFatalStroke(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()
        self._models = {}
        self._models[OutcomeModelType.DEMENTIA] = DementiaModel()
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.STROKE, 1)

    def assign_non_cv_mortality(self, person, years=1):
        return False


class AlwaysNonFatalMI(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()
        self._models = {}
        self._models[OutcomeModelType.DEMENTIA] = DementiaModel()
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return Outcome(OutcomeType.MI, 0)

    def assign_non_cv_mortality(self, person, years=1):
        return False


class AlwaysDementia(OutcomeModelRepository):
    def __init__(self):
        super(OutcomeModelRepository, self).__init__()
        self._models = {}
        self._models[OutcomeModelType.GLOBAL_COGNITIVE_PERFORMANCE] = GCPModel()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None):
        return None

    def get_dementia(self, person):
        return Outcome(OutcomeType.DEMENTIA, False)

    def assign_non_cv_mortality(self, person, years=1):
        return False


class TestQALYAssignment(unittest.TestCase):
    def initializeAfib(person):
        return None

    def getPerson(self, age=65):
        return Person(
            age=age,
            gender=0,
            raceEthnicity=1,
            sbp=140,
            dbp=80,
            a1c=6.5,
            hdl=50,
            totChol=210,
            ldl=90,
            trig=150,
            bmi=22,
            waist=50,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=1,
            alcohol=0,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine = 0,
            initializeAfib=TestQALYAssignment.initializeAfib,
            initializationRepository=InitializationRepository(),
        )

    def setUp(self):
        self._hasNoConditions = self.getPerson()
        self._hasDementia = self.getPerson()
        self._hasStroke = self.getPerson()
        self._hasFatalStroke = self.getPerson()
        self._hasMI = self.getPerson()
        self._age90 = self.getPerson(90)

    def testAgeQALYsOnly(self):
        self.assertEqual(1, self._hasNoConditions._qalys[-1])
        self.assertEqual(0.8, self._age90._qalys[-1])

    def testStrokeQALYS(self):
        self.assertEqual(1, self._hasStroke._qalys[0])
        self._hasStroke.advance_year(
            DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalStroke()
        )
        self._hasStroke.advance_year(
            DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalStroke()
        )
        self._hasStroke.advance_year(
            DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalStroke()
        )
        self.assertEqual(1, self._hasStroke._qalys[0])
        self.assertEqual(0.67, self._hasStroke._qalys[1])
        self.assertEqual(0.9, self._hasStroke._qalys[2])
        self.assertEqual(0.9, self._hasStroke._qalys[3])

    def testMIQALYS(self):
        self.assertEqual(1, self._hasMI._qalys[0])
        self._hasMI.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalMI())
        self._hasMI.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalMI())
        self._hasMI.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalMI())
        self.assertEqual(1, self._hasMI._qalys[0])
        self.assertEqual(0.88, self._hasMI._qalys[1])
        self.assertEqual(0.9, self._hasMI._qalys[2])
        self.assertEqual(0.9, self._hasMI._qalys[3])

    def testDementiaQALYS(self):
        self.assertEqual(1, self._hasDementia._qalys[0])
        self._hasDementia.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysDementia())
        self._hasDementia.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysDementia())
        self._hasDementia.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysDementia())
        self.assertEqual(1, self._hasDementia._qalys[0])
        self.assertEqual(0.80, self._hasDementia._qalys[1])
        self.assertEqual(0.79, self._hasDementia._qalys[2])
        self.assertEqual(0.78, self._hasDementia._qalys[3])

    def testQALYSWithMultipleConditions(self):
        self.assertEqual(1, self._hasMI._qalys[0])
        self._hasMI.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalMI())
        self._hasMI.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalStroke())
        self._hasMI.advance_year(DoNotChangeRiskFactorsModelRepository(), AlwaysNonFatalMI())
        self.assertEqual(1, self._hasMI._qalys[0])
        self.assertEqual(0.88, self._hasMI._qalys[1])
        # 0.9 * 0.678
        self.assertAlmostEqual(0.603, self._hasMI._qalys[2], places=5)
        # 0.9 * 0.9
        self.assertEqual(0.81, self._hasMI._qalys[3])

    def testQALYsWithDeath(self):
        self.assertEqual(1, self._hasFatalStroke._qalys[0])
        self._hasFatalStroke.advance_year(
            DoNotChangeRiskFactorsModelRepository(), AlwaysFatalStroke()
        )
        self.assertEqual(1, self._hasFatalStroke._qalys[0])
        self.assertEqual(0, self._hasFatalStroke._qalys[1])


if __name__ == "__main__":
    unittest.main()
