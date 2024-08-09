import numpy as np

from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus

class SBIModel:
    """Silent brain infarct."""
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
        
        xb = -8.2055 -0.2554 #original SBI model intercept plus the change found from recalibration

        if gender==NHANESGender.MALE:
            xb += 0.0677

        if raceEthnicity==RaceEthnicity.OTHER:
            xb += -0.00734
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.1707
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += -0.1234
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -0.0707

        if (smokingStatus!=SmokingStatus.NEVER):
            xb += 0.0501

        if statin:
            xb += 0.0138

        if afib:
            xb += 0.075

        if pvd:
            xb += 0.0843
  
        xb += age * 0.0559
        xb += sbp * 0.00308
        xb += dbp * 0.0126
        xb += bmi * (-0.0184)

        if anyPhysicalActivity:
            xb += (-0.0823)

        xb += antiHypertensiveCount*0.1337

        #if otherLipidLowering:
        #    xb += 0.00221

        xb += a1c*0.0712
        xb += totChol*0.00231
        xb += hdl*(-0.00434)
        xb += ldl*(-0.00118)
        xb += trig*(-0.00011)
        xb += creatinine*0.0917     
            
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






