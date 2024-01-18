import numpy as np
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
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
        if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
            xb += 1.23
        elif ( (raceEthnicity==NHANESRaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==NHANESRaceEthnicity.OTHER_HISPANIC) ):
            xb += 0.17
        elif raceEthnicity == NHANESRaceEthnicity.OTHER:
            xb += -1.78

        return xb
        
    def estimate_next_risk(self, person, rng=None, boolean=True):

        lp = self.calc_linear_predictor_for_patient_characteristics(
                 person._age[-1], 
                 person._sbp[-1], 
                 person._dbp[-1], 
                 person._totChol[-1], 
                 person._hdl[-1], 
                 person._gender, 
                 person._smokingStatus,
                 person._raceEthnicity)
       
        risk = np.exp(lp)/(1+np.exp(lp)) #this is a logistic model

        return rng.uniform()<risk if boolean else risk

    def estimate_next_risk_vectorized(self, x, rng=None, boolean=True):

        lp = self.calc_linear_predictor_for_patient_characteristics(
                 x.age,
                 x.sbp,
                 x.dbp,
                 x.totChol,
                 x.hdl,
                 x.gender,
                 x.smokingStatus,
                 x.raceEthnicity)
  
        risk = np.exp(lp)/(1+np.exp(lp)) #this is a logistic model

        return rng.uniform()<risk if boolean else risk

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
            if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
                xb += 1.09
            elif ( (raceEthnicity==NHANESRaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==NHANESRaceEthnicity.OTHER_HISPANIC) ):
                xb += 0.27
            elif raceEthnicity == NHANESRaceEthnicity.OTHER:
                xb += -1.34

        return xb

    def estimate_next_risk(self, person, rng=None):

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

        risk = np.exp(lp)/(1+np.exp(lp)) #this is a logistic model

        return rng.uniform()<risk

    def estimate_next_risk_vectorized(self, x, rng=None):

        lp = self.calc_linear_predictor_for_patient_characteristics(
                 x.age,
                 x.sbp,
                 x.dbp,
                 x.totChol,
                 x.hdl,
                 x.gender,
                 x.smokingStatus,
                 x.raceEthnicity,
                 x.pvd)

        risk = np.exp(lp)/(1+np.exp(lp)) #this is a logistic model

        return rng.uniform()<risk




