
import numpy as np

# TODO: this class needs to be renamed. its no longer interfacing with statsmodel
# conceptually, what it does now is bridge the regression model and the person


class StatsModelLinearRiskFactorModel:

    def __init__(self, regression_model, log_transform=False):
        self.parameters = regression_model._coefficients
        self.standard_errors = regression_model._coefficient_standard_errors
        self.residual_mean = regression_model._residual_mean
        self.residual_standard_deviation = regression_model._residual_standard_deviation
        self.log_transform = log_transform

    def get_modified_attribute_for_parameter_from_person(self, name, person):
        returnParam = self.get_modified_parameter_for_person(name, person)
        if not isinstance(returnParam, list) and not isinstance(returnParam, np.ndarray):
            return returnParam
        else:
            return returnParam[-1]

    def convert_first_letter_to_lower(self, toLower):
        return toLower[:1].lower() + toLower[1:]

    '''
    This will apply an order of operations to the elements of the mdoel name. So, meanLogLagSbp
    would take lagSBP, log it and take the mean. If the order of operations for the first elements
    matters,they'll be appleid in order of which they are listed...
    '''

    def get_modified_parameter_for_person(self, name, person):
        if name.startswith("log"):
            name = self.convert_first_letter_to_lower(name[len("log"):])
            return np.log(self.get_modified_parameter_for_person(name, person))
        elif name.startswith("mean"):
            name = self.convert_first_letter_to_lower(name[len("mean"):])
            return np.array(self.get_modified_parameter_for_person(name, person)).mean()
        # just strip lag prefixes
        elif name.startswith("lag"):
            name = self.convert_first_letter_to_lower(name[len("lag"):])
            return getattr(person, "_" + name)
        else:
            return getattr(person, "_" + name)

    def strip_categorical_name(self, name):
        stripped_name = "_" + name[:name.index("[")]
        stripped_value = int(name[name.index("[T.") + len("[T."): name.index("]")])
        return (stripped_name, stripped_value)

    def estimate_next_risk(self, person):
        # TODO: think about what to do with teh hard-coded strings for parameters and prefixes
        linearPredictor = self.parameters['Intercept']
        nonInterceptParams = dict(self.parameters)
        del nonInterceptParams['Intercept']

        # sort parametesr into categorical and non-categorial
        categoricalParams = {}
        nonCategoricalParams = {}

        for coeff_name, coeff_val in nonInterceptParams.items():
            if "[" in coeff_name:
                categoricalParams[coeff_name] = coeff_val
            else:
                nonCategoricalParams[coeff_name] = coeff_val

        # for non-categorical parameters this is easy — just add the linear predictor
        for coeff_name, coeff_val in nonCategoricalParams.items():
            linearPredictor += coeff_val * \
                self.get_modified_attribute_for_parameter_from_person(coeff_name, person)

        # for categorical params, pick which parameter to add...
        for coeff_name, coeff_val in categoricalParams.items():
            stripped_name, matched_categorical_value = self.strip_categorical_name(coeff_name)
            if (matched_categorical_value == getattr(person, stripped_name)):
                linearPredictor += coeff_val

        if (self.log_transform):
            linearPredictor = np.exp(linearPredictor)
        return linearPredictor
