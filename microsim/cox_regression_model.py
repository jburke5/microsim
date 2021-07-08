import json
from microsim.regression_model import RegressionModel


class CoxRegressionModel(RegressionModel):
    def __init__(
        self, coefficients, coefficient_standard_errors, residual_mean, residual_standard_deviation
    ):
        self._coefficients = coefficients
        self._coefficient_standard_errors = coefficient_standard_errors
        self._one_year_linear_cumulative_hazard = residual_mean
        self._one_year_quad_cumulative_hazard = residual_standard_deviation

    # will use the JSON format for Regression Model and use the one year cumulative hazard...
    def to_json(self):
        model_spec = {
            "coefficients": self._coefficients,
            "coefficient_standard_errors": self._coefficient_standard_errors,
            "residual_mean": self._one_year_linear_cumulative_hazard,
            "residual_standard_deviation": self._one_year_quad_cumulative_hazard,
        }
        return json.dumps(model_spec)
