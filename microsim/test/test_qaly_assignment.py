import unittest

from microsim.person import Person
from microsim.test.test_risk_model_repository import TestRiskModelRepository


class TestQALYAssignment(unittest.TestCase):

    def initializeAfib(person):
        return None

    def setUp(self):
        self._hasDementia = Person(
            age=75,
            gender=0,
            raceEthnicity=1,
            sbp=self.140,
            dbp=self.80,
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
            initializeAfib=TestTreatmentStrategy.initializeAfib)

        self._hasNoConditions = Person(
            age=75,
            gender=0,
            raceEthnicity=1,
            sbp=self.140,
            dbp=self.80,
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
            initializeAfib=TestTreatmentStrategy.initializeAfib)

        self._hasStroke = Person(
            age=75,
            gender=0,
            raceEthnicity=1,
            sbp=self.140,
            dbp=self.80,
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
            initializeAfib=TestTreatmentStrategy.initializeAfib)

                self._hasMI = Person(
            age=75,
            gender=0,
            raceEthnicity=1,
            sbp=self.140,
            dbp=self.80,
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
            initializeAfib=TestTreatmentStrategy.initializeAfib)


    # for  dementia... Jönsson, L., Andreasen, N., Kilander, L., Soininen, H., Waldemar, G., Nygaard, H., et al. (2006). Patient- and proxy-reported utility in Alzheimer disease using the EuroQoL. Alzheimer Disease and Associated Disorders, 20(1), 49–55. http://doi.org/10.1097/01.wad.0000201851.52707.c9
    # it has utilizites at bsaeline and with dementia follow-up...seems like a simple thing  to use...

    # for stroke/MI...Sussman, J., Vijan, S., & Hayward, R. (2013). Using benefit-based tailored treatment to improve the use of antihypertensive medications. Circulation, 128(21), 2309–2317. http://doi.org/10.1161/CIRCULATIONAHA.113.002290
    # same idea, has utilities at bsaeline and in sugsequent years...so, it shoudl be relatifely easy to use.

    def testBaselineQALYS(self):

    def testQALYSOneYearAfterEvent(self):

    def testQALYSTwoYearsAfterEvent(self):

    # for aging...Netuveli, G. (2006). Quality of life at older ages: evidence from the English longitudinal study of aging (wave 1). Journal of Epidemiology and Community Health, 60(4), 357–363. http://doi.org/10.1136/jech.2005.040071
    #  i think we can get away with something htat sets QALYS at 1 < 70 and then applies a baseline reduction of something like 10% per decade
    def testQALYSByAge(self):

    def testQALYSWithConditionAtOlderAge(self):

    def testQALYSWithMultipleConditions(self):

