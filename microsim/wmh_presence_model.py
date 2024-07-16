import numpy as np

from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.modality import Modality

class WMHPresenceCTModel:
    """White matter hypodensity model for persons with modality CT.
    This is a logistic model.
    This model, in combination with WMHPresenceMRModel, perform better at classifying persons than the 
    overall WMHPresenceModel that includes modality as an extra term."""
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

        xb = -10.6141
        if gender==NHANESGender.MALE:
            xb += 0.00675
        if raceEthnicity==RaceEthnicity.OTHER:
            xb += 0.000957
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.0868
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += -0.2014
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -0.0246
        if (smokingStatus!=SmokingStatus.NEVER):
            xb += 0.058
        if statin:
            xb += 0.0259
        if afib:
            xb += 0.0354
        if pvd:
            xb += 0.027
        xb += age*0.1084
        xb += sbp*0.00398
        xb += dbp*0.0167
        xb += bmi*(-0.00892)
        if anyPhysicalActivity:
            xb += (-0.0579)
        xb += antiHypertensiveCount*0.092
        #if otherLipidLowering:
        #    xb += 0.00221    
        xb += a1c*0.0438
        xb += totChol*(-0.0015)
        xb += hdl*0.00461
        xb += ldl*(-0.00016)
        xb += trig*0.000337
        xb += creatinine*0.1123
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
        lp = self.calc_linear_predictor_for_patient_characteristics(
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

        return True if person._rng.uniform()<self.inverse_logit(lp) else False

class WMHPresenceMRModel:
    """White matter hypodensity model for all persons with modality MR.
    This is a logistic model"""
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

        xb = -4.977
        if gender==NHANESGender.MALE:
            xb += -0.1297
        if raceEthnicity==RaceEthnicity.OTHER:
            xb += -0.0704
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += -0.0125
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += -0.0178
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += 0.0801
        if (smokingStatus!=SmokingStatus.NEVER):
            xb += 0.0641
        if statin:
            xb += 0.00244
        if afib:
            xb += 0.0341
        if pvd:
            xb += 0.0317
        xb += age*0.0618
        xb += sbp*0.00531
        xb += dbp*0.0092
        xb += bmi*(-0.0177)
        if anyPhysicalActivity:
            xb += (-0.0399)
        xb += antiHypertensiveCount*0.1089
        #if otherLipidLowering:
        #    xb += 0.00221
        xb += a1c*0.0199
        xb += totChol*(-0.00166)
        xb += hdl*0.00299
        xb += ldl*(-0.00184)
        xb += trig*0.000388
        xb += creatinine*0.0147
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
        lp = self.calc_linear_predictor_for_patient_characteristics(
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

        return True if person._rng.uniform()<self.inverse_logit(lp) else False

class WMHPresenceModel:
    """White matter hypodensity model for all persons, independently of their modality
    Modality is a term in this logistic model.
    This model predicts essentially the WMHSeverity.NO portion of the WMHSeverityModel.
    A WMHPresenceModel result as True includes the unknown WMH severity portion of the population."""
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

        xb = -8.2295
        if gender==NHANESGender.MALE:
            xb += -0.0522
        if raceEthnicity==RaceEthnicity.OTHER:
            xb += -0.0261
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.0571
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += -0.1324
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += 0.00955
        if (smokingStatus!=SmokingStatus.NEVER):
            xb += 0.0567
        if statin:
            xb += 0.00433
        if afib:
            xb += 0.0607
        if pvd:
            xb += 0.0364
        xb += age*0.0942
        xb += sbp*0.00371
        xb += dbp*0.0137
        xb += bmi*(-0.0139)
        if anyPhysicalActivity:
            xb += (-0.0656)
        xb += antiHypertensiveCount*0.1
        #if otherLipidLowering:
        #    xb += 0.00221
        xb += a1c*0.0332
        xb += totChol*(-0.00159)
        xb += hdl*0.00402
        xb += ldl*0.000676
        xb += trig*0.000334
        xb += creatinine*0.1008
        if modality==Modality.CT.value:
            xb += -0.9498
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
        lp = self.calc_linear_predictor_for_patient_characteristics(
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

        return True if person._rng.uniform()<self.inverse_logit(lp) else False     







