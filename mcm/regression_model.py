import json


class RegressionModel:
    def __init__(
            self,
            coefficients,
            coefficient_standard_errors,
            residual_mean,
            residual_standard_deviation,
    ):
        self._coefficients = coefficients
        self._coefficient_standard_errors = coefficient_standard_errors
        self._residual_mean = residual_mean
        self._residual_standard_deviation = residual_standard_deviation

    def to_json(self):
        model_spec = {
            'coefficients': self._coefficients,
            'coefficient_standard_errors': self._coefficient_standard_errors,
            'residual_mean': self._residual_mean,
            'residual_standard_deviation': self._residual_standard_deviation,
        }
        return json.dumps(model_spec)

    def write_json(self, filepath):
        with open(filepath, 'w') as model_spec_file:
            model_spec_json = self.to_json()
            model_spec_file.write(model_spec_json)
