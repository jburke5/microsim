import numpy as np
from enum import Enum

from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.modality import Modality

class WMHSeverity(Enum):
    NO = "no"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

class WMHSeverityModel():
    def __init__(self):
        pass

    def calc_linear_predictor_for_patient_characteristics(
        self,
        gender,
        raceEthnicity,
        smokingStatus,
        statin,
        afib,
        pvd,
        age,
        sbp,
        dbp,
        bmi,
        anyPhysicalActivity,
        antiHypertensiveCount,
        #otherLipidLowering,
        a1c,
        totChol,
        hdl,
        ldl,
        trig,
        creatinine,
        modality):
        """Returns the linear predictor (without the intercept) that corresponds to the cumulative probability.
        In ordered logistic regression the coefficient of each parameter is the same for all covariates (an assumption).
        So, I can calculate the linear predictor without the intercepts, because these are different for each ordered class."""

        xb = 0.

        if gender==NHANESGender.MALE:
            xb += 0.0694

        if raceEthnicity==RaceEthnicity.OTHER:
            xb += 0.00781
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += -0.0118
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += 0.1351
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -0.0158

        if (smokingStatus!=SmokingStatus.NEVER):
            xb += -0.0544

        if statin:
            xb += -0.00698

        if afib:
            xb += -0.0494

        if pvd:
            xb += -0.053

        xb += age*(-0.0903)
        xb += sbp*(-0.00266)
        xb += dbp*(-0.0157)
        xb += bmi*0.015

        if anyPhysicalActivity:
            xb += 0.0695

        xb += antiHypertensiveCount*(-0.1046)

        #if otherLipidLowering:
        #    xb += 0.00221

        xb += a1c*(-0.0296)
        xb += totChol*0.00178
        xb += hdl*(-0.00411)
        xb += ldl*(-0.00111)
        xb += trig*(-0.0004)
        xb += creatinine*(-0.0923)

        if modality==Modality.CT:
            xb += 0.9846

        return xb

    def inverse_logit(self, lp):
        if lp<-10:
            risk = 0.
        elif lp>10.:
            risk = 1.
        else:
            risk = 1/(1+np.exp(-lp))
        return risk
     
    def estimate_next_risk(self, person):
        lpWithoutIntercept = self.calc_linear_predictor_for_patient_characteristics(
            person._gender,
            person._raceEthnicity,
            person._smokingStatus,
            person._statin[-1],
            person._afib[-1],
            person._pvd[-1],
            person._age[-1],
            person._sbp[-1],
            person._dbp[-1],
            person._bmi[-1],
            person._anyPhysicalActivity[-1],
            person._antiHypertensiveCount[-1],
            #person._otherLipidLowering,
            person._a1c[-1],
            person._totChol[-1],
            person._hdl[-1],
            person._ldl[-1],
            person._trig[-1],
            person._creatinine[-1],
            person._modality)
        
        #obtain the linear predictors
        lpNoWMH = lpWithoutIntercept + 8.2116
        lpMildWMH = lpWithoutIntercept + 10.2237
        lpModerateWMH = lpWithoutIntercept + 11.6124
        #obtain the first three cumulative probabilities for the first three classes, last cumulative probability is 1.
        noWMHCumulative = self.inverse_logit(lpNoWMH)
        mildWMHCumulative = self.inverse_logit(lpMildWMH)
        moderateWMHCumulative = self.inverse_logit(lpModerateWMH)
        
        #make the decision
        draw = person._rng.uniform()
        if draw<noWMHCumulative:
            return WMHSeverity.NO
        elif draw<mildWMHCumulative:
            return WMHSeverity.MILD
        elif draw<moderateWMHCumulative:
            return WMHSeverity.MODERATE
        elif draw<1.:
            return WMHSeverity.SEVERE
        else:
            raise RuntimeError("Draw inconsistent with cumulative probabilities in WMHSeverityModel.")
