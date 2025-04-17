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

class WMHSeverityMRModel():
    """This is the WMH severity model for persons with MR as their modality.
    This is an ordered logistic model."""
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
        creatinine):
        """Returns the linear predictor (without the intercept) that corresponds to the cumulative probability.
        In ordered logistic regression the coefficient of each parameter is the same for all covariates (an assumption).
        So, I can calculate the linear predictor without the intercepts, because these are different for each ordered class."""

        xb = 0.
        if gender==NHANESGender.MALE:
            xb += 0.1329
        elif gender==NHANESGender.FEMALE:
            pass
        else:
            raise RuntimeError("unrecognized gender in WMHSeverityModel")
        if raceEthnicity==RaceEthnicity.OTHER:
            xb += 0.0569
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += -0.00583
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += 0.0206
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -0.0571
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_WHITE:
            pass
        else:
            raise RuntimeError("Unrecognized raceEthnicity in WMHSeverityModel")
        if (smokingStatus!=SmokingStatus.NEVER):
            xb += -0.0655
        if statin:
            xb += -0.00763
        if afib:
            xb += -0.0277
        if pvd:
            xb += -0.0414
        xb += age*(-0.0663)
        xb += sbp*(-0.00426)
        xb += dbp*(-0.0119)
        xb += bmi*0.018
        if anyPhysicalActivity:
            xb += 0.0494
        xb += antiHypertensiveCount*(-0.1311)
        #if otherLipidLowering:
        #    xb += 0.0513
        xb += a1c*(-0.0144)
        xb += totChol*0.00166
        xb += hdl*(-0.00344)
        xb += ldl*(-0.00198)
        xb += trig*(-0.00044)
        xb += creatinine*(-0.0177)
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
            person._creatinine[-1])

        #obtain the linear predictors
        lpNoWMH = lpWithoutIntercept + 5.4914
        lpMildWMH = lpWithoutIntercept + 7.5405
        lpModerateWMH = lpWithoutIntercept + 9.1338
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

class WMHSeverityCTModel():
    """This is the WMH severity model for persons with CT as their modality.
    This is an ordered logistic model."""
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
        creatinine):
        """Returns the linear predictor (without the intercept) that corresponds to the cumulative probability.
        In ordered logistic regression the coefficient of each parameter is the same for all covariates (an assumption).
        So, I can calculate the linear predictor without the intercepts, because these are different for each ordered class."""

        xb = 0.
        if gender==NHANESGender.MALE:
            xb += 0.00875
        elif gender==NHANESGender.FEMALE:
            pass
        else:
            raise RuntimeError("unrecognized gender in WMHSeverityModel")
        if raceEthnicity==RaceEthnicity.OTHER:
            xb += -0.0254
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += -0.0218
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += 0.2256
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += 0.0111
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_WHITE:
            pass
        else:
            raise RuntimeError("Unrecognized raceEthnicity in WMHSeverityModel")
        if (smokingStatus!=SmokingStatus.NEVER):
            xb += -0.0546
        if statin:
            xb += -0.0275
        if afib:
            xb += -0.0225
        if pvd:
            xb += -0.0455
        xb += age*(-0.1047)
        xb += sbp*(-0.00285)
        xb += dbp*(-0.0185)
        xb += bmi*0.0102
        if anyPhysicalActivity:
            xb += 0.0618
        xb += antiHypertensiveCount*(-0.0888)
        #if otherLipidLowering:
        #    xb += 0.0513
        xb += a1c*(-0.0428)
        xb += totChol*0.00185
        xb += hdl*(-0.00479)
        xb += ldl*(-0.0003)
        xb += trig*(-0.00042)
        xb += creatinine*(-0.1055)
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
            person._creatinine[-1])

        #obtain the linear predictors
        lpNoWMH = lpWithoutIntercept + 10.6378
        lpMildWMH = lpWithoutIntercept + 12.5635
        lpModerateWMH = lpWithoutIntercept + 13.7202
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

class WMHSeverityModel():
    """This is the WMH severity model for all persons, independently of their modality.
    Modality is a term in this model. This is an ordered logistic model.
    This model was found to classify WMH outcomes a bit worse than the WMHSeverityCTModel and WMHSeverityMRModel on the Kaiser population."""
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
        elif gender==NHANESGender.FEMALE:
            pass
        else:
            raise RuntimeError("unrecognized gender in WMHSeverityModel")
        if raceEthnicity==RaceEthnicity.OTHER:
            xb += 0.00781
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += -0.0118
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += 0.1351
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -0.0158
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_WHITE:
            pass
        else:
            raise RuntimeError("Unrecognized raceEthnicity in WMHSeverityModel")
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
        #    xb += 0.0513
        xb += a1c*(-0.0296)
        xb += totChol*0.00178
        xb += hdl*(-0.00411)
        xb += ldl*(-0.00111)
        xb += trig*(-0.0004)
        xb += creatinine*(-0.0923)
        if modality==Modality.CT.value:
            xb += 0.9846
        elif modality==Modality.MR.value:
            pass
        elif modality==Modality.NO.value:
            pass
        else:
            raise RuntimeError("Unrecognized modality in WMHSeverityModel")
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
        #these are the sum of two terms, the intercept as obtained from the original fit, and one obtained from our own recalibration
        lpNoWMH = lpWithoutIntercept + 8.2116 -0.26733
        lpMildWMH = lpWithoutIntercept + 10.2237 -0.43271
        lpModerateWMH = lpWithoutIntercept + 11.6124 -0.49049
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
