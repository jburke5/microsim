import unittest
import copy
import numpy as np

from microsim.person import Person
from microsim.test.test_risk_model_repository import TestRiskModelRepository
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.smoking_status import SmokingStatus
from microsim.outcome import Outcome, OutcomeType
from microsim.alcohol_category import AlcoholCategory
from microsim.gcp_model import GCPModel


def initializeAfib(arg):
    return False


def initializeAfibAlwaysPositive(arg):
    return True


class AlwaysFatalStrokeOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysFatalStrokeOutcomeRepository, self).__init__()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None, rng=None):
        return Outcome(OutcomeType.STROKE, True)

    def assign_non_cv_mortality(self, person, rng=None):
        return False

    def get_random_effects(self):
        return {}

    def get_gcp(self, person, rng=None):
        return GCPModel().get_risk_for_person(person, test=True)

    def get_gcp_vectorized(self, person):
        return GCPModel().get_risk_for_person(person, vectorized=True, test=True)


class AlwaysNonCVDeathRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysNonCVDeathRepository, self).__init__()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None, rng=None):
        return None

    def assign_non_cv_mortality(self, person, rng=None):
        return True

    def get_random_effects(self):
        return {}

    def get_gcp(self, person, rng=None):
        return GCPModel().get_risk_for_person(person, test=True)

    def get_gcp_vectorized(self, person):
        return GCPModel().get_risk_for_person(person, vectorized=True, test=True)


class AlwaysNonFatalStrokeOutcomeRepository(OutcomeModelRepository):
    def __init__(self):
        super(AlwaysNonFatalStrokeOutcomeRepository, self).__init__()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None, rng=None):
        return Outcome(OutcomeType.STROKE, False)

    def assign_non_cv_mortality(self, person, rng=None):
        return False

    def get_random_effects(self):
        return {}

    def get_gcp(self, person, rng=None):
        return GCPModel().get_risk_for_person(person, test=True)

    def get_gcp_vectorized(self, person):
        return GCPModel().get_risk_for_person(person, vectorized=True, test=True)


class NothingHappensRepository(OutcomeModelRepository):
    def __init__(self):
        super(NothingHappensRepository, self).__init__()

    def assign_cv_outcome(self, person, years=1, manualStrokeMIProbability=None, rng=None):
        return None

    def assign_non_cv_mortality(self, person, rng=None):
        return False

    def get_random_effects(self):
        return {}

    def get_gcp(self, person, rng=None):
        return GCPModel().get_risk_for_person(person, test=True)

    def get_gcp_vectorized(self, person):
        return GCPModel().get_risk_for_person(person, vectorized=True, test=True)


class TestResetPersonAndRollBackEvents(unittest.TestCase):
    def setUp(self):
        self.baseAge = 55
        self.baseSBP = 120
        self._white_male = Person(
            age=self.baseAge,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_copy_paste = Person(
            age=self.baseAge,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_one_year = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_female = Person(
            age=self.baseAge,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._black_female = Person(
            age=self.baseAge,
            gender=NHANESGender.FEMALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_BLACK,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_sbp = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP + 10,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_dbp = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=90,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_a1c = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=7,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_hdl = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=60,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_totChol = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=223,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_ldl = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=100,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_trig = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=160,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_bmi = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=25,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_waist = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=36,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_activity = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=1,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_minus_edudcation = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=1,
            education=Education.HIGHSCHOOLGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_smoking = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=1,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.CURRENT,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_bpMed = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=1,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_statin = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=1,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_lipid = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=1,
            creatinine=0.0,
            initializeAfib=initializeAfib,
        )

        self._white_male_plus_afib = Person(
            age=self.baseAge + 1,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfibAlwaysPositive,
        )

        self._baseline_stroke_person = Person(
            age=self.baseAge,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
            selfReportStrokeAge=50,
        )

        self._baseline_stroke_person_copy_paste = Person(
            age=self.baseAge,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=self.baseSBP,
            dbp=80,
            a1c=6,
            hdl=50,
            totChol=213,
            ldl=90,
            trig=150,
            bmi=22,
            waist=34,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.NEVER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            creatinine=0.0,
            initializeAfib=initializeAfib,
            selfReportStrokeAge=50,
        )

    def testResetBasicAttributes(self):
        self._white_male.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self._baseline_stroke_person.advance_year(
            TestRiskModelRepository(), AlwaysFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(2, len(self._white_male._dbp))

        self._white_male.reset_to_baseline()

        self.assertEqual(1, len(self._white_male._dbp))
        self.assertEqual(1, len(self._white_male._totChol))
        self.assertEqual(1, len(self._white_male._trig))
        self.assertEqual(1, len(self._white_male._trig))
        self.assertEqual(1, len(self._white_male._sbp))

        self.assertEqual(self.baseSBP, self._white_male._sbp[-1])
        self.assertEqual(self.baseAge, self._white_male._age[-1])

    def testResetOutcomes(self):
        self._white_male.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self._baseline_stroke_person.advance_year(
            TestRiskModelRepository(), AlwaysFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(1, len(self._white_male._outcomes[OutcomeType.STROKE]))
        self._white_male.reset_to_baseline()
        self.assertEqual(0, len(self._white_male._outcomes[OutcomeType.STROKE]))

    def testResetOutcomesPreservesPreSimOutcomes(self):
        self._white_male.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self._baseline_stroke_person.advance_year(
            TestRiskModelRepository(), AlwaysFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(2, len(self._baseline_stroke_person._outcomes[OutcomeType.STROKE]))
        self._baseline_stroke_person.reset_to_baseline()
        self.assertEqual(1, len(self._baseline_stroke_person._outcomes[OutcomeType.STROKE]))

    def testRollbackNonFatalEvent(self):
        self._white_male.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self._baseline_stroke_person.advance_year(
            TestRiskModelRepository(), AlwaysFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(False, self._white_male.has_stroke_prior_to_simulation())
        self.assertEqual(True, self._white_male.has_stroke_during_simulation())
        self.assertEqual(True, self._white_male.has_stroke_during_wave(1))

        self._white_male.rollback_most_recent_event(OutcomeType.STROKE)

        self.assertEqual(False, self._white_male.has_stroke_prior_to_simulation())
        self.assertEqual(False, self._white_male.has_stroke_during_wave(1))
        self.assertEqual(False, self._white_male.has_stroke_during_simulation())

    def testRollbackFatalEvent(self):
        self._white_male.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self._baseline_stroke_person.advance_year(
            TestRiskModelRepository(), AlwaysFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(True, self._baseline_stroke_person.has_stroke_during_simulation())
        self.assertEqual(True, self._baseline_stroke_person.has_stroke_prior_to_simulation())
        self.assertEqual(True, self._baseline_stroke_person.has_stroke_during_wave(1))
        self.assertEqual(True, self._baseline_stroke_person.is_dead())

        self._baseline_stroke_person.rollback_most_recent_event(OutcomeType.STROKE)

        self.assertEqual(True, self._baseline_stroke_person.has_stroke_prior_to_simulation())
        self.assertEqual(False, self._baseline_stroke_person.has_stroke_during_wave(1))
        self.assertEqual(False, self._baseline_stroke_person.has_stroke_during_simulation())
        self.assertEqual(False, self._baseline_stroke_person.is_dead())
        self.assertEqual(self.baseAge + 1, self._baseline_stroke_person._age[-1])

    def testBasePatientEquals(self):
        self.assertTrue(self._white_male == self._white_male_copy_paste)
        self.assertEqual(self._white_male, self._white_male_copy_paste)
        self.assertNotEqual(self._white_male, self._white_female)
        self.assertNotEqual(self._white_female, self._black_female)
        self.assertNotEqual(self._white_male, self._white_male_minus_edudcation)
        self.assertNotEqual(self._white_male, self._white_male_plus_a1c)
        self.assertNotEqual(self._white_male, self._white_male_plus_activity)
        self.assertNotEqual(self._white_male, self._white_male_plus_afib)
        self.assertNotEqual(self._white_male, self._white_male_plus_bmi)
        self.assertNotEqual(self._white_male, self._white_male_plus_dbp)
        self.assertNotEqual(self._white_male, self._white_male_plus_hdl)
        self.assertNotEqual(self._white_male, self._white_male_plus_ldl)
        self.assertNotEqual(self._white_male, self._white_male_plus_lipid)
        self.assertNotEqual(self._white_male, self._white_male_plus_one_year)
        self.assertNotEqual(self._white_male, self._white_male_plus_sbp)
        self.assertNotEqual(self._white_male, self._white_male_plus_smoking)
        self.assertNotEqual(self._white_male, self._white_male_plus_statin)
        self.assertNotEqual(self._white_male, self._white_male_plus_totChol)
        self.assertNotEqual(self._white_male, self._white_male_plus_trig)
        self.assertNotEqual(self._white_male, self._white_male_plus_waist)

    def testBaselineEqualityAfterAdvancingAYear(self):
        self._white_male.advance_year(TestRiskModelRepository(), NothingHappensRepository(), rng = np.random.default_rng())
        self._white_male_copy_paste.advance_year(
            TestRiskModelRepository(), NothingHappensRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(self._white_male, self._white_male_copy_paste)

    def testBasePatientWithBaselineStroke(self):
        self.assertNotEqual(self._white_male, self._baseline_stroke_person)
        self.assertEqual(self._baseline_stroke_person, self._baseline_stroke_person_copy_paste)
        self._baseline_stroke_person.advance_year(
            TestRiskModelRepository(), NothingHappensRepository(), rng = np.random.default_rng()
        )
        self._baseline_stroke_person_copy_paste.advance_year(
            TestRiskModelRepository(), NothingHappensRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(self._baseline_stroke_person, self._baseline_stroke_person_copy_paste)

    def testBasePatientWithCVEvents(self):
        self.assertEqual(self._white_male_copy_paste, self._white_male)
        self._white_male.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertNotEqual(self._white_male, self._white_male_copy_paste)
        self._white_male_copy_paste.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(self._white_male_copy_paste, self._white_male)

    def testBasePatientWithNonCVDeath(self):
        self.assertEqual(self._white_male_copy_paste, self._white_male)
        self._white_male.advance_year(TestRiskModelRepository(), AlwaysNonCVDeathRepository(), rng = np.random.default_rng())
        self.assertNotEqual(self._white_male, self._white_male_copy_paste)
        self._white_male_copy_paste.advance_year(
            TestRiskModelRepository(), NothingHappensRepository(), rng = np.random.default_rng()
        )
        self.assertNotEqual(self._white_male_copy_paste, self._white_male)

    def testBasePatientWithNonCVDeathOtherWay(self):
        self.assertEqual(self._white_male_copy_paste, self._white_male)
        self._white_male.advance_year(TestRiskModelRepository(), AlwaysNonCVDeathRepository(), rng = np.random.default_rng())
        self.assertNotEqual(self._white_male, self._white_male_copy_paste)
        self._white_male_copy_paste.advance_year(
            TestRiskModelRepository(), AlwaysNonCVDeathRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(self._white_male_copy_paste, self._white_male)

    def testDeepCopy(self):
        self.assertEqual(self._white_male, copy.deepcopy(self._white_male))
        self._white_male.advance_year(TestRiskModelRepository(), NothingHappensRepository(), rng = np.random.default_rng())
        self.assertEqual(self._white_male, copy.deepcopy(self._white_male))
        self._white_male.advance_year(
            TestRiskModelRepository(), AlwaysNonFatalStrokeOutcomeRepository(), rng = np.random.default_rng()
        )
        self.assertEqual(self._white_male, copy.deepcopy(self._white_male))
        self._white_male.advance_year(TestRiskModelRepository(), AlwaysNonCVDeathRepository(), rng = np.random.default_rng())
        self.assertEqual(self._white_male, copy.deepcopy(self._white_male))
