from functools import reduce
import numpy as np
from mcm.model_argument_transform import get_all_argument_transforms

# TODO: this class needs to be renamed. its no longer interfacing with statsmodel
# conceptually, what it does now is bridge the regression model and the person


class StatsModelLinearRiskFactorModel:
    def __init__(self, regression_model, log_transform=False):
        self.standard_errors = regression_model._coefficient_standard_errors
        self.log_transform = log_transform
        if (
            hasattr(regression_model, "_residual_mean")
            and hasattr(regression_model, "_residual_standard_deviation")
        ):
            self.residual_mean = regression_model._residual_mean
            self.residual_standard_deviation = regression_model._residual_standard_deviation

        self.parameters = {**(regression_model._coefficients)}
        self.non_intercept_params = {k: v for k, v in self.parameters.items() if k != 'Intercept'}
        self.argument_transforms = get_all_argument_transforms(self.non_intercept_params.keys())

    def draw_from_residual_distribution(self):
        if not hasattr(self, "residual_mean") and hasattr(self, "residual_standard_deviation"):
            raise RuntimeError("Cannot draw from residual distribution: model does not have"
                               " residual information")
        return np.random.normal(
            loc=self.residual_mean,
            scale=self.residual_standard_deviation,
            size=1)[0]

    def get_intercept(self):
        return self.parameters['Intercept']

    def estimate_next_risk(self, person):
        # TODO: think about what to do with teh hard-coded strings for parameters and prefixes
        linearPredictor = self.get_intercept()

        for coeff_name, coeff_val in self.non_intercept_params.items():
            if coeff_name not in self.argument_transforms:
                model_argument = getattr(person, f"_{coeff_name}")
            else:
                prop_name, transforms = self.argument_transforms[coeff_name]
                prop_value = getattr(person, f"_{prop_name}")
                model_argument = reduce(lambda v, f: f(v), transforms, prop_value)
            if isinstance(model_argument, list) or isinstance(model_argument, np.ndarray):
                model_argument = model_argument[-1]
            linearPredictor += coeff_val * model_argument

        if (self.log_transform):
            linearPredictor = np.exp(linearPredictor)

        return linearPredictor
