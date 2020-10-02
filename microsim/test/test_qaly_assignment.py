import unittest

from microsim.person import Person
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.education import Education
from microsim.outcome import Outcome, OutcomeType


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
            initializeAfib=TestQALYAssignment.initializeAfib)
    
    def setUp(self):
        self._hasNoConditions = self.getPerson()
        self._hasDementia = self.getPerson()
        self._hasDementia.add_outcome_event(Outcome(OutcomeType.DEMENTIA, False))

        self._hasStroke = self.getPerson()
        self._hasStroke.add_outcome_event(Outcome(OutcomeType.STROKE, False))

        self._hasFatalStroke = self.getPerson()
        self._hasFatalStroke.add_outcome_event(Outcome(OutcomeType.STROKE, True))
        
        self._hasMI = self.getPerson()
        self._hasMI.add_outcome_event(Outcome(OutcomeType.MI, True))

        self._age90 = self.getPerson(90)

    # for  dementia... Jönsson, L., Andreasen, N., Kilander, L., Soininen, H., Waldemar, G., Nygaard, H., et al. (2006). 
    # Patient- and proxy-reported utility in Alzheimer disease using the EuroQoL. Alzheimer Disease and Associated Disorders, 
    # 20(1), 49–55. http://doi.org/10.1097/01.wad.0000201851.52707.c9
    # it has utilizites at bsaeline and with dementia follow-up...seems like a simple thing  to use...

    # for stroke/MI...Sussman, J., Vijan, S., & Hayward, R. (2013). Using benefit-based tailored treatment to improve the use of 
    # antihypertensive medications. Circulation, 128(21), 2309–2317. http://doi.org/10.1161/CIRCULATIONAHA.113.002290
    # same idea, has utilities at bsaeline and in sugsequent years...so, it shoudl be relatifely easy to use.

    # for aging...Netuveli, G. (2006). Quality of life at older ages: evidence from the English longitudinal study of aging (wave 1). 
    # Journal of Epidemiology and Community Health, 60(4), 357–363. http://doi.org/10.1136/jech.2005.040071
    #  i think we can get away with something htat sets QALYS at 1 < 70 and then applies a baseline reduction of 
    # something like 10% per decade


    def testBaselineQALYS(self):
        self.assertEqual(1, self._hasNoConditions._qalys[-1])
        self.assertEqual(1, self._hasDementia._qalys[-1])
        self.assertEqual(1, self._hasStroke._qalys[-1])
        self.assertEqual(1, self._hasMI._qalys[-1])
        self.assertEqual(1, self._hasFatalStroke._qalys[-1])
        self.assertEqual(0.8, self._age90._qalys[-1])

    def testQALYSOneYearAfterEvent(self):
        pass

    def testQALYSTwoYearsAfterEvent(self):
        pass

    def testQALYSByAge(self):
        pass

    def testQALYSWithConditionAtOlderAge(self):
        pass

    def testQALYSWithMultipleConditions(self):
        pass
