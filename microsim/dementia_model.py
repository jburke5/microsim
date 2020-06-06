from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel


class DementiaModel(StatsModelCoxModel):

    def __init__(self):
        super(DementiaModel, self).__init__(CoxRegressionModel({}, {}, 1.15252068e-05, 2.58682684e-06), False)
        # fit slope in notebook lookAtSurvivalFunctionForDementiaModel

    def linear_predictor(self, person):
        xb = 0
        xb += person._age[-1] * 0.10237
        xb += person._gcp[0] * -0.07549

        slope = 0
        # can only calculate slope for people under observation for 2 or more years...
        if len(person._gcp) >= 2:
            slope = person._gcp[-1] - person._gcp[-2]
        xb += slope * -0.00100

        if person._gender == NHANESGender.MALE:
            xb += -0.09506

        if person._education == Education.LESSTHANHIGHSCHOOL:
            xb += 0.03109
        elif person._education == Education.SOMEHIGHSCHOOL:
            xb += 0.08430
        elif person._education == Education.HIGHSCHOOLGRADUATE:
            xb += -0.08468
        elif person._education == Education.SOMECOLLEGE:
            xb += -0.22637

        if person._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.19377
        return xb
