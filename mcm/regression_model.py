import json


class RegressionModel:
    def __init__(self, params, ses, residual_mean, residual_sd):
        self._params = params
        self._ses = ses
        self._residual_mean = residual_mean
        self._residual_sd = residual_sd

    def to_json(self):
        model_spec = {
            'params': self._params,
            'ses': self._ses,
            'residual_mean': self._residual_mean,
            'residual_sd': self._residual_sd,
        }
        return json.dumps(model_spec)

    def write_json(self, filepath):
        with open(filepath, 'w') as model_spec_file:
            model_spec_json = self.to_json()
            json.dump(model_spec_json, model_spec_file)
