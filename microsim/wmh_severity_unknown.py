import numpy as np
from enum import Enum

from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.modality import Modality

#class WMHSeverityUnknown(Enum):
#    UNKNOWN = "unknown"
#    KNOWN = "known"

class WMHSeverityUnknownModel:
    """White matter hypodensity severity unknown model."""
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

        xb = -8.1153

        if gender==NHANESGender.MALE:
            xb += -0.00119

        if raceEthnicity==RaceEthnicity.OTHER:
            xb += -0.00891
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.2251
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += -0.1197
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -0.053

        if (smokingStatus!=SmokingStatus.NEVER):
            xb += 0.0478

        if statin:
            xb += -0.00178

        if afib:
            xb += 0.0536

        if pvd:
            xb += -0.014
   
        xb += age*0.0704
        xb += sbp*0.00327
        xb += dbp*0.00656
        xb += bmi*(-0.00913)

        if anyPhysicalActivity:
            xb += (-0.0503)

        xb += antiHypertensiveCount*0.0582

        #if otherLipidLowering:
        #    xb += -0.0133

        xb += a1c*0.0264
        xb += totChol*0.000864
        xb += hdl*0.000177
        xb += ldl*(-0.00192)
        xb += trig*(-0.00023)
        xb += creatinine*0.0867

        if modality==Modality.CT.value:
            xb += -0.2186

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







