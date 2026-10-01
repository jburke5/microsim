from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.risk_factors.race_ethnicity import RaceEthnicity


class NHANESLinearRiskFactorModel:

    """
    Predicts next risk factor for a Person by applying a linear regression. Every known risk factor
    on a Person should be included in a risk factor model to ensure that coerrelations between
    risk factors are maintained across time.
    """

    def __init__(self, params, resids):

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

        self._resids = resids

    def estimate_risk_for_params(self, age, gender, sbp, dbp, a1c, hdl, totChol, bmi, raceEthnicity, smokingStatus, rng=None):
        linear_pred = 0
        linear_pred += age * self._params["age"]
        linear_pred += gender * self._params["gender"]
        linear_pred += sbp * self._params["sbp"]
        linear_pred += dbp * self._params["dbp"]
        linear_pred += a1c * self._params["a1c"]
        linear_pred += hdl * self._params["hdl"]
        linear_pred += totChol * self._params["totChol"]
        linear_pred += bmi * self._params["bmi"]
        linear_pred += self._params["intercept"]

        if raceEthnicity == RaceEthnicity.OTHER_HISPANIC:
            linear_pred += self._params["raceEth2"]
        elif (raceEthnicity == RaceEthnicity.NON_HISPANIC_WHITE) | (raceEthnicity == RaceEthnicity.ASIAN):
            linear_pred += self._params["raceEth3"]
        elif raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
            linear_pred += self._params["raceEth4"]
        elif raceEthnicity == RaceEthnicity.OTHER:
            linear_pred += self._params["raceEth5"]

        if smokingStatus == SmokingStatus.FORMER:
            linear_pred += self._params["smokingStatus1"]
        elif smokingStatus == SmokingStatus.CURRENT:
            linear_pred += self._params["smokingStatus2"]

        linear_pred += rng.normal(self._resids.mean(), self._resids.std())

        return self.transform_linear_predictor(linear_pred)

    def estimate_next_risk(self, person):
        return self.estimate_risk_for_params(age=person._age[-1], gender=person._gender, sbp=person._sbp[-1],
            dbp=person._dbp[-1], a1c=person._a1c[-1], hdl=person._hdl[-1], totChol=person._totChol[-1], bmi=person._bmi[-1],
            raceEthnicity=person._raceEthnicity, smokingStatus=person._smokingStatus, rng=person._rng)


    """A stub method so that sub-classes can override to transform the risks """

    def transform_linear_predictor(self, linear_pred):
        return linear_pred
