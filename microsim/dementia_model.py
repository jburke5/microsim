import numpy as np

from microsim.race_ethnicity import RaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel
from microsim.outcome import OutcomeType, Outcome
from microsim.modality import Modality
from microsim.wmh_severity import WMHSeverity

class DementiaModel(StatsModelCoxModel):

    # initial parameters in notebook lookAtSurvivalFunctionForDementiaModel (linearTerm=1.33371239e-05, quadraticTerm=5.64485841e-05)
    # recalibrated fit to population incidence equation in notebook: identifyOptimalBaselineSurvivalParametersForDementia, linear multiplier = 0.5, quad = 0.05

    def __init__(
        self, linearTerm=1.33371239e-05, quadraticTerm=5.64485841e-05, wmhSpecific=True, populationRecalibration=True
    ):
        super().__init__(CoxRegressionModel({}, {}, linearTerm, quadraticTerm), False)
        if populationRecalibration:
            self.one_year_linear_cumulative_hazard = self.one_year_linear_cumulative_hazard * 0.5
            self.one_year_quad_cumulative_hazard = self.one_year_quad_cumulative_hazard * 0.175
        self.wmhSpecific = wmhSpecific

    def generate_next_outcome(self, person):
        fatal = False
        return Outcome(OutcomeType.DEMENTIA, fatal)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person) if person._rng.uniform(size=1)<self.get_risk_for_person(person, years=1) else None

    def linear_predictor(self, person):
        return self.linear_predictor_for_patient_characteristics(
            currentAge=person._age[-1],
            baselineGcp=person._baselineGcp,
            gcpSlope=person._gcpSlope,
            gender=person._gender,
            education=person._education,
            raceEthnicity=person._raceEthnicity,
            modality=person._modality,
            sbi=person.get_outcome_item_first(OutcomeType.WMH, "sbi", inSim=True),
            wmh=person.get_outcome_item_first(OutcomeType.WMH, "wmh", inSim=True),
            severityUnknown=person.get_outcome_item_first(OutcomeType.WMH, "wmhSeverityUnknown", inSim=True),
            severity=person.get_outcome_item_first(OutcomeType.WMH, "wmhSeverity", inSim=True),
        )

    def linear_predictor_for_patient_characteristics(
        self, currentAge, baselineGcp, gcpSlope, gender, education, raceEthnicity, modality, sbi, wmh, severityUnknown, severity
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

        if raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.1937563

        if self.wmhSpecific: #if we just want a mean increased risk for the kaiser population then the modified linear and quadratic term adjustment did it    
            if sbi:
                if currentAge < 70:
                    xb += np.log(2.02)
                else:
                    xb += np.log(1.22) 
            if modality == Modality.MR.value:
                if severityUnknown:
                    xb += np.log(1.67)
                elif severity == WMHSeverity.MILD:
                    xb += np.log(1.41)
                elif severity == WMHSeverity.MODERATE:
                    xb += np.log(2.03)
                elif severity == WMHSeverity.SEVERE:
                    xb += np.log(2.32)
            elif modality == Modality.CT.value:
                if severityUnknown:
                    xb += np.log(3.40)
                elif severity == WMHSeverity.MILD:
                    xb += np.log(2.62)
                elif severity == WMHSeverity.MODERATE:
                    xb += np.log(4.16)
                elif severity == WMHSeverity.SEVERE:
                    xb += np.log(4.11)
                elif severity == WMHSeverity.NO:
                    xb += np.log(1.58)

        return xb

