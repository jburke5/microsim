from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel


class DementiaModel(StatsModelCoxModel):

    def __init__(self):
        super(DementiaModel, self).__init__(CoxRegressionModel({}, {}, 1.33371239e-05, 5.64485841e-05), False)
        # fit slope in notebook lookAtSurvivalFunctionForDementiaModel

    def linear_predictor(self, person):
        xb = 0
        xb += person._age[-1] * 0.1023685
        xb += person._gcp[0] * -0.0754936

        # can only calculate slope for people under observation for 2 or more years...
        slope = 0
        if len(person._gcp) >= 2:
            slope = person._gcp[-1] - person._gcp[-2]
        xb += slope * -0.000999

        if person._gender == NHANESGender.FEMALE:
            xb += 0.0950601

        if person._education == Education.LESSTHANHIGHSCHOOL:
            xb += 0.0307459
        elif person._education == Education.SOMEHIGHSCHOOL:
            xb += 0.0841255
        elif person._education == Education.HIGHSCHOOLGRADUATE:
            xb += -0.0846951
        elif person._education == Education.SOMECOLLEGE:
            xb += -0.2263593

        if person._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.1937563
        return xb
