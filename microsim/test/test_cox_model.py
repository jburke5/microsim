from microsim.cox_regression_model import CoxRegressionModel
from microsim.data_loader import load_model_spec
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.test.fixture.vectorized_test_fixture import VectorizedTestFixture


class TestCoxModel(VectorizedTestFixture):
    def setUp(self):
        super().setUp()
        model_spec = load_model_spec("nhanesMortalityModel")
        self.model = StatsModelCoxModel(CoxRegressionModel(**model_spec))

    def test_single_linear_predictor(self):
        # Baseline estimate derived in notebook — buildNHANESMortalityModel.
        # only testing to 3 places because we approximate the cumulative hazard as oppossed
        # as opposed to directly using it
        p1_data = self.population_dataframe.iloc[0]

        actual_current_risk = self.model.linear_predictor_vectorized(p1_data)
        actual_cumulative_risk = self.model.get_risk_for_person(p1_data, years=1, vectorized=True,)

        self.assertAlmostEqual(
            first=5.440096345569454, second=actual_current_risk, places=1,
        )
        self.assertAlmostEqual(
            first=0.026299703075722214, second=actual_cumulative_risk, places=1,
        )
