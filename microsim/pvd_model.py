import numpy as np
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender

# based on the publication: https://doi.org/10.1097%2FMD.0000000000003454
# the models, both prevalence and incidence, produce results quantitatively different from, but qualitatively similar to, the GBD data
# perhaps a small adjustment may be done in the future
class PVDPrevalenceModel:

    def  __init__(self):
        pass

    def calc_linear_predictor_for_patient_characteristics(
        self,
        age,
        sbp,
        dbp,
        totChol,
        hdl,
        gender,
        smokingStatus,
        raceEthnicity):

        ageS = age/10.
        pulsePressureS = (sbp-dbp)/10.
        tchdlRatio = totChol/hdl

        xb = -9.37
        xb += 0.69*ageS
        xb += 0.17*pulsePressureS
        xb += 0.10*tchdlRatio
        if gender == NHANESGender.FEMALE:
            xb += 0.14
        if ( (smokingStatus==SmokingStatus.CURRENT) | (smokingStatus==SmokingStatus.FORMER) ):
            xb += 0.47
        #it seems that white was the reference, so if we want to map asian to white for the microsim models we should not change this code...
        if raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 1.23
        elif ( (raceEthnicity== RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC) ):
            xb += 0.17
        elif raceEthnicity == RaceEthnicity.OTHER:
            xb += -1.78

        return xb
        
    def estimate_next_risk(self, person, boolean=True):

        lp = self.calc_linear_predictor_for_patient_characteristics(
                 person._age[-1], 
                 person._sbp[-1], 
                 person._dbp[-1], 
                 person._totChol[-1], 
                 person._hdl[-1], 
                 person._gender, 
                 person._smokingStatus,
                 person._raceEthnicity)
       
        # note: this is a logistic model
        # note: limit the calculation to avoid over/under-flow issues
        if lp<-10:
            risk = 0.
        elif lp>10.:
            risk = 1.
        else:
            risk = 1/(1+np.exp(-lp))

        return person._rng.uniform()<risk if boolean else risk

# developed using the PVD prevalence model above, see pvdModelDevelopment notebooks for details
class PVDIncidenceModel:
    def  __init__(self):
        pass

    def calc_linear_predictor_for_patient_characteristics(
        self,
        age,
        sbp,
        dbp,
        totChol,
        hdl,
        gender,
        smokingStatus,
        raceEthnicity,
        lagPVD):

        # this ensures that if someone gets PVD, they will continue having it
        # in the future, we could adjust this to include misdiagnosis etc
        if lagPVD:
            xb = 10
        else:
            ageS = age/10.
            pulsePressureS = (sbp-dbp)/10.
            tchdlRatio = totChol/hdl

            xb = -11.41
            xb += 0.95*ageS
            xb += -0.25*pulsePressureS
            xb += 0.07*tchdlRatio
            if gender == NHANESGender.FEMALE:
                xb += 0.01
            if ( (smokingStatus==SmokingStatus.CURRENT) | (smokingStatus==SmokingStatus.FORMER) ):
                xb += 0.02
            if raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
                xb += 1.09
            elif ( (raceEthnicity== RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC) ):
                xb += 0.27
            elif raceEthnicity == RaceEthnicity.OTHER:
                xb += -1.34

        return xb

    def estimate_next_risk(self, person):

        lp = self.calc_linear_predictor_for_patient_characteristics(
                 person._age[-1],
                 person._sbp[-1],
                 person._dbp[-1],
                 person._totChol[-1],
                 person._hdl[-1],
                 person._gender,
                 person._smokingStatus,
                 person._raceEthnicity,
                 person._pvd[-1])

        # note: this is a logistic model
        # note: limit the calculation to avoid over/under-flow issues
        if lp<-10:
            risk = 0.
        elif lp>10.:
            risk = 1.
        else:
            risk = 1/(1+np.exp(-lp))

        return person._rng.uniform()<risk





