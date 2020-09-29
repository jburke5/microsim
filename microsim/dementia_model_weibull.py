import numpy as np

from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel


class DementiaModelWeibull(StatsModelCoxModel):

    # initial parameters fit to population incidence equation in notebook: identifyOptimalBaselineSurvivalParametersForDementia
    def __init__(self, linearTerm=0.0, quadraticTerm=0.0):
        super().__init__(
            CoxRegressionModel({}, {}, linearTerm, quadraticTerm), False)
        self.shapeParameter = 0.242891669
        self.scalingFactor = 1
        self.ageFactor =  0.175429647
        # fit slope in notebook lookAtSurvivalFunctionForDementiaModel

    def linear_predictor(self, person):
        xb = 0
        xb += person._age[-1] *  0.175429647
        xb += person._gcp[0] * -0.060809704

        # can only calculate slope for people under observation for 2 or more years...
        slope = 0
        if len(person._gcp) >= 2:
            slope = person._gcp[-1] - person._gcp[-2]
        xb += slope * -0.001158854

        if person._gender == NHANESGender.FEMALE:
            xb += 0.091511933

        # less than high school is default, 0
        if person._education == Education.SOMEHIGHSCHOOL:
            xb += 0.141510702
        elif person._education == Education.HIGHSCHOOLGRADUATE:
            xb += 0.046780821
        elif person._education == Education.SOMECOLLEGE:
            xb += -0.007563779
        elif person._education == Education.COLLEGEGRADUATE:
            xb += 0.117845198

        
        raceCoeff =-0.213494525
        if person._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += raceCoeff
        elif person._raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_WHITE:
            xb += raceCoeff*2


        xb += -15.274047241 # intercept
        return xb
    
    def get_cumulative_hazard_for_lp(self, lp, time):
        return np.exp(lp/self.scalingFactor) / self.shapeParameter * (np.exp(self.shapeParameter * time) - 1)

    def get_cumulative_hazard(self, person, time):
        return self.get_cumulative_hazard_for_lp(self.linear_predictor(person), time)
            
    def get_risk_for_person(self, person, years):
        return self.get_cumulative_hazard(person, len(person._age)) - self.get_cumulative_hazard(person, len(person._age) - 1)

