from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel


class DementiaModel(StatsModelCoxModel):

    # initial parameters in notebook lookAtSurvivalFunctionForDementiaModel (linearTerm=1.33371239e-05, quadraticTerm=5.64485841e-05)
    # recalibrated fit to population incidence equation in notebook: identifyOptimalBaselineSurvivalParametersForDementia, linear multiplier = 0.5, quad = 0.05

    def __init__(
        self, linearTerm=1.33371239e-05, quadraticTerm=5.64485841e-05, populationRecalibration=True
    ):
        super().__init__(CoxRegressionModel({}, {}, linearTerm, quadraticTerm), False)
        if populationRecalibration:
            self.one_year_linear_cumulative_hazard = self.one_year_linear_cumulative_hazard * 0.5
            self.one_year_quad_cumulative_hazard = self.one_year_quad_cumulative_hazard * 0.05

    def linear_predictor(self, person):
        return self.linear_predictor_for_patient_characteristics(
            currentAge=person._age[-1],
            baselineGcp=person._gcp[0],
            gcpSlope=person._gcp[-1] - person._gcp[-2] if len(person._gcp) >= 2 else 0,
            gender=person._gender,
            education=person._education,
            raceEthnicity=person._raceEthnicity,
        )

    def linear_predictor_for_patient_characteristics(
        self, currentAge, baselineGcp, gcpSlope, gender, education, raceEthnicity
    ):
        xb = 0
        xb += currentAge * 0.1023685
        xb += baselineGcp * -0.0754936

        # can only calculate slope for people under observation for 2 or more years...
        xb += gcpSlope * -0.000999

        if gender == NHANESGender.FEMALE:
            xb += 0.0950601

        if education == Education.LESSTHANHIGHSCHOOL:
            xb += 0.0307459
        elif education == Education.SOMEHIGHSCHOOL:
            xb += 0.0841255
        elif education == Education.HIGHSCHOOLGRADUATE:
            xb += -0.0846951
        elif education == Education.SOMECOLLEGE:
            xb += -0.2263593

        if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.1937563
        return xb

    # need to override for specific subclasses that implement it.
    def linear_predictor_vectorized(self, x):
        return self.linear_predictor_for_patient_characteristics(
            currentAge=x.age,
            baselineGcp=x.baseGcp,
            gcpSlope=x.gcpSlope,
            gender=x.gender,
            education=x.education,
            raceEthnicity=x.raceEthnicity,
        )
