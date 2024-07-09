import numpy as np

from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.modality import Modality

class WMHPresenceModel:
    """White matter hypodensity model."""
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

        if modality==Modality.CT:
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







