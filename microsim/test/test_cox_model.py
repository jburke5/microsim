import unittest

from microsim.person import Person
from microsim.gender import NHANESGender
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.cox_regression_model import CoxRegressionModel
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.education import Education
from microsim.data_loader import load_model_spec
from microsim.alcohol_category import AlcoholCategory


def initializeAfib(person):
    return None


class TestCoxModel(unittest.TestCase):
    def setUp(self):
        self.imputed_dataset_first_person = Person(
            age=71,
            gender=NHANESGender.MALE,
            raceEthnicity=NHANESRaceEthnicity.NON_HISPANIC_WHITE,
            sbp=144.667,
            dbp=52.6667,
            a1c=9.5,
            hdl=34,
            totChol=191,
            bmi=30.05,
            ldl=110.0,
            trig=128,
            waist=45,
            anyPhysicalActivity=0,
            education=Education.COLLEGEGRADUATE,
            smokingStatus=SmokingStatus.FORMER,
            alcohol=AlcoholCategory.NONE,
            antiHypertensiveCount=0,
            statin=0,
            otherLipidLoweringMedicationCount=0,
            initializeAfib=initializeAfib,
        )

        model_spec = load_model_spec("nhanesMortalityModel")
        self.model = StatsModelCoxModel(CoxRegressionModel(**model_spec))

    def test_single_linear_predictor(self):
        # Baseline estimate derived in notebook — buildNHANESMortalityModel.
        # only testing to 3 places because we approximate the cumulative hazard as oppossed
        # as opposed to directly using it
        actual_current_risk = self.model.linear_predictor_vectorized(
            self.imputed_dataset_first_person
        )
        actual_cumulative_risk = self.model.get_risk_for_person(
            self.imputed_dataset_first_person,
            years=1,
            vectorized=True
        ),

        self.assertAlmostEqual(
            first=5.440096345569454,
            second=actual_current_risk,
            places=1,
        )
        self.assertAlmostEqual(
            first=0.026299703075722214,
            second=actual_cumulative_risk,
            places=1,
        )


if __name__ == "__main__":
    try:
        TestCoxModel(methodName='test_single_linear_predictor').debug()
    except Exception:
        import pdb, sys, traceback
        errtype, errvalue, tb = sys.exc_info()
        traceback.print_exception(errtype, errvalue, tb)
        pdb.post_mortem(tb)
