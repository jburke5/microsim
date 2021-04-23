import unittest

from microsim.alcohol_category import AlcoholCategory
from microsim.cox_regression_model import CoxRegressionModel
from microsim.data_loader import load_model_spec
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.person import Person
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.smoking_status import SmokingStatus
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.test.helper.init_vectorized_population_dataframe import (
    init_vectorized_population_dataframe
)


class TestCoxModel(unittest.TestCase):
    def setUp(self):
        self.population_dataframe = init_vectorized_population_dataframe([
            Person(
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
                initializeAfib=(lambda _: None),
                randomEffects={'gcp': 0},
            ),
        ], with_base_gcp=True)
        model_spec = load_model_spec("nhanesMortalityModel")
        self.model = StatsModelCoxModel(CoxRegressionModel(**model_spec))

    def test_single_linear_predictor(self):
        # Baseline estimate derived in notebook — buildNHANESMortalityModel.
        # only testing to 3 places because we approximate the cumulative hazard as oppossed
        # as opposed to directly using it
        p1_data = self.population_dataframe.iloc[0]

        actual_current_risk = self.model.linear_predictor_vectorized(p1_data)
        actual_cumulative_risk = self.model.get_risk_for_person(
            p1_data,
            years=1,
            vectorized=True,
        )

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
