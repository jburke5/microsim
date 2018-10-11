import numpy as np


class LinearRiskFactorModel:
    """
    Predicts next risk factor for a Person by applying a linear regression. Every known risk factor
    on a Person should be included in a risk factor model to ensure that coerrelations between
    risk factors are maintained across time.
    """

    def __init__(self, name, params, ses):
        # i'm sure there is a more elegant way to do this...
        self._params = {
            'age': params['age'],
            'gender': params['gender'],
            'raceEth2': params['raceEthnicity[T.2]'],
            'raceEth3': params['raceEthnicity[T.3]'],
            'raceEth4': params['raceEthnicity[T.4]'],
            'raceEth5': params['raceEthnicity[T.5]'],
            'sbp': params['sbp'],
            'dbp': params['dbp'],
            'a1c': params['a1c'],
            'hdl': params['hdl'],
            'tot_chol': params['tot_chol'],
            'intercept': params['Intercept'],
        }
        self._ses = {
            'age': ses['age'],
            'gender': ses['gender'],
            'raceEth2': ses['raceEthnicity[T.2]'],
            'raceEth3': ses['raceEthnicity[T.3]'],
            'raceEth4': ses['raceEthnicity[T.4]'],
            'raceEth5': ses['raceEthnicity[T.5]'],
            'sbp': ses['sbp'],
            'dbp': ses['dbp'],
            'a1c': ses['a1c'],
            'hdl': ses['hdl'],
            'tot_chol': ses['tot_chol'],
            'intercept': ses['Intercept'],
        }

    def get_coefficent_from_params(self, param):
        return np.random.normal(self._params[param], self._ses[param])

    def estimate_next_risk(self, age, gender, race_ethnicity, sbp, dbp, a1c, hdl, chol):
        linear_pred = 0
        linear_pred += age * self.get_coefficent_from_params('age')
        linear_pred += gender * self.get_coefficent_from_params('gender')
        linear_pred += sbp * self.get_coefficent_from_params('sbp')
        linear_pred += dbp * self.get_coefficent_from_params('dbp')
        linear_pred += a1c * self.get_coefficent_from_params('a1c')
        linear_pred += hdl * self.get_coefficent_from_params('hdl')
        linear_pred += chol * self.get_coefficent_from_params('tot_chol')
        linear_pred += self.get_coefficent_from_params('intercept')

        if (race_ethnicity == 2):
            linear_pred += self.get_coefficent_from_params('raceEth2')
        elif (race_ethnicity == 3):
            linear_pred += self.get_coefficent_from_params('raceEth3')
        elif (race_ethnicity == 4):
            linear_pred += self.get_coefficent_from_params('raceEth4')
        elif (race_ethnicity == 5):
            linear_pred += self.get_coefficent_from_params('raceEth5')

        return self.transform_linear_predictor(linear_pred)

    '''A stub method so that sub-classes can override to transform the risks '''

    def transform_linear_predictor(self, linear_pred):
        return linear_pred
