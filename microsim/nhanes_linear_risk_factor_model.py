from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity

import numpy as np


class NHANESLinearRiskFactorModel:

    """
    Predicts next risk factor for a Person by applying a linear regression. Every known risk factor
    on a Person should be included in a risk factor model to ensure that coerrelations between
    risk factors are maintained across time.
    """

    def __init__(self, name, params, ses, resids):

        # i'm sure there is a more elegant way to do this...
        self._params = {
            "age": params["age"],
            "gender": params["gender"],
            "raceEth2": params["raceEthnicity[T.2]"],
            "raceEth3": params["raceEthnicity[T.3]"],
            "raceEth4": params["raceEthnicity[T.4]"],
            "raceEth5": params["raceEthnicity[T.5]"],
            "smokingStatus1": params["smokingStatus[T.1]"],
            "smokingStatus2": params["smokingStatus[T.2]"],
            "sbp": params["sbp"],
            "dbp": params["dbp"],
            "a1c": params["a1c"],
            "hdl": params["hdl"],
            "totChol": params["totChol"],
            "bmi": params["bmi"],
            "intercept": params["Intercept"],
        }
        self._ses = {
            "age": ses["age"],
            "gender": ses["gender"],
            "raceEth2": ses["raceEthnicity[T.2]"],
            "raceEth3": ses["raceEthnicity[T.3]"],
            "raceEth4": ses["raceEthnicity[T.4]"],
            "raceEth5": ses["raceEthnicity[T.5]"],
            "smokingStatus1": ses["smokingStatus[T.1]"],
            "smokingStatus2": ses["smokingStatus[T.2]"],
            "sbp": ses["sbp"],
            "dbp": ses["dbp"],
            "a1c": ses["a1c"],
            "hdl": ses["hdl"],
            "totChol": ses["totChol"],
            "bmi": ses["bmi"],
            "intercept": ses["Intercept"],
        }

        self._resids = resids

    def get_coefficent_from_params(self, param):
        return np.random.normal(self._params[param], self._ses[param])

    def estimate_risk_for_params(self, age, gender, sbp, dbp, a1c, hdl, totChol, bmi, raceEthnicity, smokingStatus):
        linear_pred = 0
        linear_pred += age * self.get_coefficent_from_params("age")
        linear_pred += gender * self.get_coefficent_from_params("gender")
        linear_pred += sbp * self.get_coefficent_from_params("sbp")
        linear_pred += dbp * self.get_coefficent_from_params("dbp")
        linear_pred += a1c * self.get_coefficent_from_params("a1c")
        linear_pred += hdl * self.get_coefficent_from_params("hdl")
        linear_pred += totChol * self.get_coefficent_from_params("totChol")
        linear_pred += bmi * self.get_coefficent_from_params("bmi")
        linear_pred += self.get_coefficent_from_params("intercept")

        if raceEthnicity == NHANESRaceEthnicity.OTHER_HISPANIC:
            linear_pred += self.get_coefficent_from_params("raceEth2")
        elif raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_WHITE:
            linear_pred += self.get_coefficent_from_params("raceEth3")
        elif raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            linear_pred += self.get_coefficent_from_params("raceEth4")
        elif raceEthnicity == NHANESRaceEthnicity.OTHER:
            linear_pred += self.get_coefficent_from_params("raceEth5")

        if smokingStatus == SmokingStatus.FORMER:
            linear_pred += self.get_coefficent_from_params("smokingStatus1")
        elif smokingStatus == SmokingStatus.CURRENT:
            linear_pred += self.get_coefficent_from_params("smokingStatus2")

        linear_pred += np.random.normal(self._resids.mean(), self._resids.std())

        return self.transform_linear_predictor(linear_pred)

    def estimate_next_risk(self, person):
        return self.estimate_risk_for_params(age=person._age[-1], gender=person._gender, sbp=person._sbp[-1],
            dbp=person._dbp[-1], a1c=person._a1c[-1], hdl=person._hdl[-1], totChol=person._totChol[-1], bmi=person._bmi[-1], 
            raceEthnicity=person._raceEthnicity, smokingStatus=person._smokingStatus)

    def estimate_next_risk_vectorized(self, x):
        #print(list(x.index))
        return self.estimate_risk_for_params(age=x.age, gender=x.gender, sbp=x.sbp,
            dbp=x.dbp, a1c=x.a1c, hdl=x.hdl, totChol=x.totChol, bmi=x.bmi, 
            raceEthnicity=x.raceEthnicity, smokingStatus=x.smokingStatus)
    
    """A stub method so that sub-classes can override to transform the risks """

    def transform_linear_predictor(self, linear_pred):
        return linear_pred
