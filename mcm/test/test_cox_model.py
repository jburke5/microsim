import unittest
import os
import json

from mcm.person import Person
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.cox_regression_model import CoxRegressionModel
from mcm.statsmodel_cox_model import StatsModelCoxModel
from mcm.education import Education


def initializeAFib(person):
    return None


class TestCoxModel(unittest.TestCase):
    def setUp(self):
        self.imputed_dataset_first_person = Person(71, NHANESGender.MALE,
                                                   NHANESRaceEthnicity.NON_HISPANIC_WHITE,
                                                   144.667, 52.6667, 9.5, 34, 191, 30.05,
                                                   110.0, 128, 45, 0, Education.COLLEGEGRADUATE,
                                                   SmokingStatus.FORMER, 0, 0, 0,
                                                   initializeAFib)

        abs_module_path = os.path.abspath(os.path.dirname(__file__))
        model_spec_path = os.path.normpath(os.path.join(
            abs_module_path, "../data/", "nhanesMortalityModelSpec.json"))
        with open(model_spec_path, 'r') as model_spec_file:
            model_spec = json.load(model_spec_file)
        self.model = StatsModelCoxModel(CoxRegressionModel(**model_spec))

    def test_single_linear_predictor(self):
        # Baseline estimate derived in notebook — buildNHANESMortalityModel.
        # only testing to 3 places because we approximate the cum hazard as oppossed
        # as opposed to directly using it
        self.assertAlmostEqual(
            first=5.440096345569454, second=self.model.linear_predictor(
                self.imputed_dataset_first_person), places=1)
        self.assertAlmostEqual(
            first=0.026299703075722214, second=self.model.get_risk_for_person(
                self.imputed_dataset_first_person, 1), places=1)


if __name__ == "__main__":
    unittest.main()
