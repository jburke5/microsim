import numpy as np

from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.regression_models.cox_risk_factor_model import CoxRiskFactorModel
from microsim.regression_models.cox_regression_model import CoxRegressionModel
from microsim.outcomes.outcome import OutcomeType, Outcome
from microsim.outcomes.outcome_prevalence_base import OutcomePrevalenceBase
from microsim.risk_factors.modality import Modality
from microsim.outcomes.wmh_severity import WMHSeverity
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType

class DementiaModel(CoxRiskFactorModel):

    # initial parameters in notebook lookAtSurvivalFunctionForDementiaModel (linearTerm=1.33371239e-05, quadraticTerm=5.64485841e-05)
    # recalibrated fit to population incidence equation in notebook: identifyOptimalBaselineSurvivalParametersForDementia, linear multiplier = 0.5, quad = 0.175

    def __init__(
        self, linearTerm=1.33371239e-05, quadraticTerm=5.64485841e-05, wmhSpecific=True, populationRecalibration=True, riskScaling=1.0
    ):
        super().__init__(CoxRegressionModel({}, {}, linearTerm, quadraticTerm), False)
        self._riskScaling = riskScaling
        if populationRecalibration:
            self.one_year_linear_cumulative_hazard = self.one_year_linear_cumulative_hazard * 0.5
            self.one_year_quad_cumulative_hazard = self.one_year_quad_cumulative_hazard * 0.175
        self.wmhSpecific = wmhSpecific

    def generate_next_outcome(self, person):
        fatal = False
        return Outcome(OutcomeType.DEMENTIA, fatal)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person) if person._rng.uniform(size=1)<self.get_risk_for_person(person, years=1) else None

    def get_risk_for_person(self, person, years=1):
        risk = super().get_risk_for_person(person, years=years)

        risk = risk * self._riskScaling

        tst = TreatmentStrategiesType.WMD15.value
        if "wmd15MedsAdded" in person._treatmentStrategies[tst]:
            wmd15MedsAdded = person._treatmentStrategies[tst]['wmd15MedsAdded']
            risk = risk * 0.85 if wmd15MedsAdded>0 else risk

        tst = TreatmentStrategiesType.WMD20.value
        if "wmd20MedsAdded" in person._treatmentStrategies[tst]:
            wmd20MedsAdded = person._treatmentStrategies[tst]['wmd20MedsAdded']
            risk = risk * 0.775 if wmd20MedsAdded>0 else risk

        tst = TreatmentStrategiesType.WMD25.value
        if "wmd25MedsAdded" in person._treatmentStrategies[tst]:
            wmd25MedsAdded = person._treatmentStrategies[tst]['wmd25MedsAdded']
            risk = risk * 0.71 if wmd25MedsAdded>0 else risk
        return risk

    def linear_predictor(self, person):
        return self.linear_predictor_for_patient_characteristics(
            currentAge=person._age[-1],
            baselineGcp=person._baselineGcp,
            gcpSlope=person._gcpSlope,
            gender=person._gender,
            education=person._education,
            raceEthnicity=person._raceEthnicity,
            modality=person._modality,
            hasBrainScan=person.has_brain_scan(),
            sbi=person.get_outcome_item_first(OutcomeType.WMH, "sbi", inSim=True),
            wmh=person.get_outcome_item_first(OutcomeType.WMH, "wmh", inSim=True),
            severityUnknown=person.get_outcome_item_first(OutcomeType.WMH, "wmhSeverityUnknown", inSim=True),
            severity=person.get_outcome_item_first(OutcomeType.WMH, "wmhSeverity", inSim=True),
        )

    def linear_predictor_for_patient_characteristics(
        self, currentAge, baselineGcp, gcpSlope, gender, education, raceEthnicity, modality, hasBrainScan, sbi, wmh, severityUnknown, severity
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

        if self.wmhSpecific and hasBrainScan: #if we just want a mean increased risk for the kaiser population then the modified linear and quadratic term adjustment did it
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


class DementiaPrevalenceModel(OutcomePrevalenceBase):
    """Logistic prevalence model that seeds priorToSim dementia at Person construction.
       Coefficients below are placeholder zeros — replace with fitted odds ratios.
       The placeholder intercept is -12, not 0: an intercept of 0 with all-zero coefficients
       seeds dementia at expit(0)=50% in every default population. At -12 the default seeding
       is ~0 (expit(-12)~6e-6) while riskScaling still shifts the odds, so calibrate_prevalence
       can drive prevalence to a target within its logS bracket of [-15, 15]."""

    _outcomeType = OutcomeType.DEMENTIA

    def __init__(self, riskScaling=1.0):
        self._intercept = -12.
        self._riskScaling = riskScaling

    def get_linear_predictor_for_person(self, person):
        return self.calc_linear_predictor_for_patient_characteristics(
            person._age[-1],
            person._gender,
            person._raceEthnicity,
            person._education,
            person._smokingStatus,
            person._anyPhysicalActivity[-1],
            person._sbp[-1],
            person._dbp[-1],
            person._totChol[-1],
        )

    def calc_linear_predictor_for_patient_characteristics(
        self,
        age,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        anyPhysicalActivity,
        sbp,
        dbp,
        totChol,
    ):
        xb = self._intercept

        if age < 65:
            xb += 0.
        elif 65 <= age < 70:
            xb += 0.  # reference
        elif 70 <= age < 75:
            xb += 0.
        elif 75 <= age < 80:
            xb += 0.
        elif age >= 80:
            xb += 0.

        if gender == NHANESGender.FEMALE:
            xb += 0.
        elif gender == NHANESGender.MALE:
            xb += 0.  # reference

        if raceEthnicity == RaceEthnicity.NON_HISPANIC_WHITE:
            xb += 0.  # reference
        elif raceEthnicity == RaceEthnicity.ASIAN:
            xb += 0.
        elif raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.
        elif (raceEthnicity == RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity == RaceEthnicity.OTHER_HISPANIC):
            xb += 0.
        elif raceEthnicity == RaceEthnicity.OTHER:
            xb += 0.

        if (education == Education.LESSTHANHIGHSCHOOL) | (education == Education.SOMEHIGHSCHOOL):
            xb += 0.  # reference
        elif education == Education.HIGHSCHOOLGRADUATE:
            xb += 0.
        elif education == Education.SOMECOLLEGE:
            xb += 0.
        elif education == Education.COLLEGEGRADUATE:
            xb += 0.

        if smokingStatus == SmokingStatus.NEVER:
            xb += 0.  # reference
        elif smokingStatus == SmokingStatus.FORMER:
            xb += 0.
        elif smokingStatus == SmokingStatus.CURRENT:
            xb += 0.

        xb += anyPhysicalActivity * 0.
        xb += sbp * 0.
        xb += dbp * 0.
        xb += totChol * 0.

        return xb
