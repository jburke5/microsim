import unittest
import os
import json

from mcm.person import Person
from mcm.gender import NHANESGender
from mcm.race_ethnicity import NHANESRaceEthnicity
from mcm.smoking_status import SmokingStatus
from mcm.cox_regression_model import CoxRegressionModel
from mcm.statsmodel_cox_model import StatsModelCoxModel


class TestCoxModel(unittest.TestCase):
    def setUp(self):
        self.imputed_dataset_first_person = Person(22, NHANESGender.MALE,
                                                   NHANESRaceEthnicity.NON_HISPANIC_WHITE,
                                                   110.666667, 74.666667, 5.1, 41.0, 168.0, 23.3,
                                                   110.0, 84.0, SmokingStatus.NEVER)

        abs_module_path = os.path.abspath(os.path.dirname(__file__))
        model_spec_path = os.path.normpath(os.path.join(
            abs_module_path, "../data/", "nhanesMortalityModelSpec.json"))
        with open(model_spec_path, 'r') as model_spec_file:
            model_spec = json.load(model_spec_file)
        self.model = StatsModelCoxModel(CoxRegressionModel(**model_spec))

    def test_single_linear_predictor(self):
        self.assertAlmostEqual(
            0.0006064897336953675, self.model.get_risk_for_person(
                self.imputed_dataset_first_person, 1))


if __name__ == "__main__":
    unittest.main()
