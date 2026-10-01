import numpy as np
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.education import Education
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.outcomes.outcome import OutcomeType, Outcome
from microsim.outcomes.outcome_prevalence_base import OutcomePrevalenceBase

class EpilepsyPrevalenceModel(OutcomePrevalenceBase):
    _outcomeType = OutcomeType.EPILEPSY

    def __init__(self, riskScaling=1.0):
        self._intercept = 4.742451144
        self._riskScaling = riskScaling

    def get_risk_for_person(self, person):
        # Coefficients below are calibrated to a linear-predictor / 1000 scaling, not expit(lp).
        # Override exists to preserve that scaling; the base class default would apply expit instead.
        # Risk scaling applied as a direct rate multiplier (not odds-shift) for the same reason.
        return self.get_linear_predictor_for_person(person) / 1000. * self._riskScaling

    def get_linear_predictor_for_person(self, person):
        return self.calc_linear_predictor_for_patient_characteristics(
                   person._age[-1],
                   person._gender,
                   person._raceEthnicity,
                   person._education,
                   person._smokingStatus,
                   person._bmi[-1],
                   person._totChol[-1],
                   person._ldl[-1],
                   person._stroke,
                   person._mi,
                   person.has_diabetes(),
                   person._any_antiHypertensive,
                   person._current_ckd)

    def calc_linear_predictor_for_patient_characteristics(
        self,
        age,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        bmi,
        totChol,
        ldl,
        stroke,
        mi,
        diabetes,
        hypertension,
        ckd
        ):

        xb=self._intercept

        if age<65:
            xb += 0.676314428
        elif 65<=age<70.:
            xb += 0.  #reference
        elif 70<=age<75.:
            xb += -0.628514168
        elif 75<=age<80:
            xb += -2.246069374
        elif age>=80:
            xb += -1.688572164

        if gender==NHANESGender.FEMALE:
            xb += 2.636652047
        elif gender==NHANESGender.MALE:
            xb += 0. #reference

        if raceEthnicity==RaceEthnicity.NON_HISPANIC_WHITE:
            xb += 0. #reference
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -5.956367017
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 3.508378787
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += -0.06222887
        elif raceEthnicity==RaceEthnicity.OTHER:
            pass #coefficient was not provided for this race category

        if education==Education.HIGHSCHOOLGRADUATE:
            xb += -1.686709019
        elif education==Education.SOMECOLLEGE:
            xb += -2.549087017
        elif education==Education.COLLEGEGRADUATE:
            xb += -2.059398803
        elif (education==Education.LESSTHANHIGHSCHOOL) | (education==Education.SOMEHIGHSCHOOL):
            xb += 0. #reference

        if smokingStatus==SmokingStatus.FORMER:
            xb += -1.118085516
        elif smokingStatus==SmokingStatus.CURRENT:
            xb += 3.872519597
        elif smokingStatus==SmokingStatus.NEVER:
            xb += 0. #reference

        if bmi<25:
            xb += 0. # reference
        elif 25<=bmi<30:
            xb += -0.401691517
        elif bmi>=30:
            xb += 0.885935373

        if (totChol>240) | (ldl>190):
            xb += 0.476490103

        if stroke:
            xb += 20.34422415
        if mi: 
            xb += 2.112535203
        if diabetes:
            xb += 1.400930279
        if hypertension:
            xb += 0.47792047
        if ckd:
            xb += 1.139188756

        return xb


class EpilepsyIncidenceModel():
    def __init__(self, riskScaling=1.0):
        self._cbhfSlope = 0.001286 #fitted to first 7 years of data
        self._cbhfIntercept = 0.
        self._riskScaling = riskScaling

    def get_cumulative_baseline_hazard(self, t):
        '''Returns the cumulative baseline hazard H(0..t), with t in years since simulation start.'''
        return self._cbhfIntercept + self._cbhfSlope * t

    def get_linear_predictor_for_person(self, person):
        return self.calc_linear_predictor_for_patient_characteristics(
                   person._age[-1],
                   person._gender,
                   person._raceEthnicity,
                   person._education,
                   person._smokingStatus,
                   person._bmi[-1],
                   person._totChol[-1],
                   person._ldl[-1],
                   person._stroke,
                   person._mi,
                   person.has_diabetes(),
                   person._any_antiHypertensive,
                   person._current_ckd)

    def calc_linear_predictor_for_patient_characteristics(
        self,
        age,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        bmi,
        totChol,
        ldl,
        stroke,
        mi,
        diabetes,
        hypertension,
        ckd
        ):
        
        xb=0

        if age<65:
            xb += -0.615632618
        elif 65<=age<70.:
            xb += 0.  #reference
        elif 70<=age<75.:
            xb += 0.067645724
        elif 75<=age<80:
            xb += 0.436865346
        elif age>=80:
            xb += 0.653011494

        if gender==NHANESGender.FEMALE:
            xb += -0.098552244
        elif gender==NHANESGender.MALE:
            xb += 0. #reference

        if raceEthnicity==RaceEthnicity.NON_HISPANIC_WHITE:
            xb += 0. #reference
        elif raceEthnicity==RaceEthnicity.ASIAN:
            xb += -0.000313101
        elif raceEthnicity==RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.408860315
        elif (raceEthnicity==RaceEthnicity.MEXICAN_AMERICAN) | (raceEthnicity==RaceEthnicity.OTHER_HISPANIC):
            xb += 0.301202791
        elif raceEthnicity==RaceEthnicity.OTHER:
            pass #coefficient was not provided for this race category

        if education==Education.HIGHSCHOOLGRADUATE:
            xb += -0.06286513
        elif education==Education.SOMECOLLEGE:
            xb += -0.076452851
        elif education==Education.COLLEGEGRADUATE:
            xb += -0.0986296
        elif (education==Education.LESSTHANHIGHSCHOOL) | (education==Education.SOMEHIGHSCHOOL):
            xb += 0. #reference
 
        if smokingStatus==SmokingStatus.FORMER:
            xb += 0.058863126
        elif smokingStatus==SmokingStatus.CURRENT:
            xb += 0.331718144
        elif smokingStatus==SmokingStatus.NEVER:
            xb += 0. #reference
    
        if bmi<25:    
            xb += 0. # reference
        elif 25<=bmi<30:
            xb += -0.026455472
        elif bmi>=30:
            xb += -0.064983803

        if (totChol>240) | (ldl>190):
            xb += -0.047287917

        if stroke:
            xb += 0.687675942 
        if mi:
            xb += 0.121327841
        if diabetes:
            xb += 0.352379596
        if hypertension:
            xb += 0.153015053
        if ckd:
            xb += 0.257707409
            
        return xb


    def get_risk_for_person(self, person):
        if person.has_epilepsy(): #if a person had epilepsy in the past they will always have it
            risk = 1.
        else:
            #during the first round of outcome risk calculations waveCompleted is -1, which is the end of year 1
            t = person._waveCompleted + 2
            #conditional first-onset probability P(T<=t | T>t-1); the unconditional CDF P(T<=t) would recount years already survived
            hazardIncrement = self.get_cumulative_baseline_hazard(t) - self.get_cumulative_baseline_hazard(t-1)
            risk = 1. - np.exp( - hazardIncrement * np.exp( self.get_linear_predictor_for_person(person) ) )
            risk = risk * self._riskScaling
        return risk

    def generate_next_outcome(self, person):
        fatal = False
        return Outcome(OutcomeType.EPILEPSY, fatal)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person) if person._rng.uniform(size=1)<self.get_risk_for_person(person) else None
