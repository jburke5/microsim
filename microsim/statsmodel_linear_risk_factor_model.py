from functools import reduce
import numpy as np
from microsim.model_argument_transform import get_all_argument_transforms

# TODO: this class needs to be renamed. its no longer interfacing with statsmodel
# conceptually, what it does now is bridge the regression model and the person

INTERACTION_INDICATOR = "#"


class StatsModelLinearRiskFactorModel:
    def __init__(self, regression_model, log_transform=False):
        self.standard_errors = regression_model._coefficient_standard_errors
        self.log_transform = log_transform
        if hasattr(regression_model, "_residual_mean") and hasattr(
            regression_model, "_residual_standard_deviation"
        ):
            self.residual_mean = regression_model._residual_mean
            self.residual_standard_deviation = regression_model._residual_standard_deviation

        self.parameters = {**(regression_model._coefficients)}
        self.non_intercept_params = {k: v for k, v in self.parameters.items() if k != "Intercept"}
        self.argument_transforms = get_all_argument_transforms(self.get_keys_for_transforms())
        self.argument_transforms_vectorized = get_all_argument_transforms(
            self.get_keys_for_transforms(), True
        )

    # method to be overriden by models that want to, in addition to the risks estimated by
    # the regression coefficients loaded from a model, also be able to apply some manual parameters.
    def get_manual_parameters(self):
        return {}

    def get_keys_for_transforms(self):
        keysForTransforms = []
        for key in self.non_intercept_params.keys():
            if self.contains_interaction(key):
                keysForTransforms.extend(self.get_interactions(key))
            else:
                keysForTransforms.append(key)
        return keysForTransforms

    def draw_from_residual_distribution(self, rng=None):
        #rng = np.random.default_rng(rng)
        if not hasattr(self, "residual_mean") and hasattr(self, "residual_standard_deviation"):
            raise RuntimeError(
                "Cannot draw from residual distribution: model does not have"
                " residual information"
            )
        return rng.normal(
            loc=self.residual_mean, scale=self.residual_standard_deviation, size=1
        )[0]

    def get_intercept(self):
        return self.parameters["Intercept"]

    def get_model_argument_for_coeff_name(self, coeff_name, person):
        if coeff_name not in self.argument_transforms:
            model_argument = getattr(person, f"_{coeff_name}")
        else:
            prop_name, transforms = self.argument_transforms[coeff_name]
            prop_value = getattr(person, f"_{prop_name}")
            if isinstance(prop_value, list) or isinstance(prop_value, np.ndarray):
                prop_value = prop_value[-1]
            model_argument = reduce(lambda v, t: t.apply(v), transforms, prop_value)
        if isinstance(model_argument, list) or isinstance(model_argument, np.ndarray):
            model_argument = model_argument[-1]
        return model_argument

    def contains_interaction(self, coeff_name):
        return INTERACTION_INDICATOR in coeff_name

    def get_interactions(self, coeff_name):
        return coeff_name.split(INTERACTION_INDICATOR)

    def get_risk_for_person(self, person, rng, years, vectorized=False):
        linear_predictor = (
            self.estimate_next_risk_vectorized(person)
            if vectorized
            else self.estimate_next_risk(person)
        )
        return linear_predictor

    def estimate_next_risk(self, person, rng=None, withResidual=False):
        # TODO: think about what to do with teh hard-coded strings for parameters and prefixes
        linearPredictor = self.get_intercept()

        for coeff_name, coeff_val in self.non_intercept_params.items():
            if self.contains_interaction(coeff_name):
                interactions = []
                for interact in self.get_interactions(coeff_name):
                    interactions.append(self.get_model_argument_for_coeff_name(interact, person))
                model_argument = reduce(lambda x, y: x * y, interactions, 1)
            else:
                model_argument = self.get_model_argument_for_coeff_name(coeff_name, person)

            linearPredictor += coeff_val * model_argument

        for coeff_name, manual_tuple in self.get_manual_parameters().items():
            # the tuple gives one item as the regression coefficent and the second item as a method
            # to get the values from a person
            linearPredictor += manual_tuple[0] * manual_tuple[1](person)

        if self.log_transform:
            linearPredictor = np.exp(linearPredictor)

        return linearPredictor+self.draw_from_residual_distribution(rng=rng) if withResidual else linearPredictor

    def estimate_next_risk_vectorized(self, x, rng=None, withResidual=False):

        # TODO: think about what to do with teh hard-coded strings for parameters and prefixes
        linearPredictor = self.get_intercept()

        for coeff_name, coeff_val in self.non_intercept_params.items():
            if self.contains_interaction(coeff_name):
                interactions = []
                for interact in self.get_interactions(coeff_name):
                    interactions.append(
                        self.get_model_argument_for_coeff_name_vectorized(interact, x)
                    )
                model_argument = reduce(lambda x, y: x * y, interactions, 1)
            else:
                model_argument = self.get_model_argument_for_coeff_name_vectorized(coeff_name, x)

            linearPredictor += coeff_val * model_argument

        for coeff_name, manual_tuple in self.get_manual_parameters(True).items():
            # the tuple gives one item as the regression coefficent and the second item as a method
            # to get the values from a person
            linearPredictor += manual_tuple[0] * manual_tuple[1](x)

        if self.log_transform:
            linearPredictor = np.exp(linearPredictor)

        return linearPredictor+self.draw_from_residual_distribution(rng=rng) if withResidual else linearPredictor

    def get_model_argument_for_coeff_name_vectorized(self, coeff_name, x):
        if coeff_name not in self.argument_transforms_vectorized:
            model_argument = x[coeff_name]
        else:
            prop_name, transforms = self.argument_transforms_vectorized[coeff_name]
            for transform in list(transforms):
                transform.prop_name = prop_name
            model_argument = reduce(lambda v, t: t.apply(v), transforms, x)
        return model_argument
