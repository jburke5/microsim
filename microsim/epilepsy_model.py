import numpy as np
from microsim.race_ethnicity import RaceEthnicity
from microsim.gender import NHANESGender
from microsim.education import Education
from microsim.smoking_status import SmokingStatus
from microsim.statsmodel_cox_model import StatsModelCoxModel
from microsim.cox_regression_model import CoxRegressionModel
from microsim.outcome import OutcomeType, Outcome

class EpilepsyModel(StatsModelCoxModel):
    def __init__(
        self, linearTerm=1.33371239e-05, quadraticTerm=5.64485841e-05, populationRecalibration=True
    ):
        super().__init__(CoxRegressionModel({}, {}, linearTerm, quadraticTerm))
        if populationRecalibration:
            self.one_year_linear_cumulative_hazard = self.one_year_linear_cumulative_hazard * 0.5
            self.one_year_quad_cumulative_hazard = self.one_year_quad_cumulative_hazard * 0.175

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
        gfr
        ):
        
        xb=0

        if age<65:
            xb += np.log(0.8)
        elif 65<=age<74.9:
            xb += np.log(1)
        elif 75<=age<84.9:
            xb += np.log(1.2)
        elif age>=85:
            xb += np.log(1.5)

        if gender==NHANESGender.FEMALE:
            xb += np.log(0.9)

        if raceEthnicity!=RaceEthnicity.NON_HISPANIC_WHITE:
            xb += np.log(1.3)
   
        if (education==Education.SOMEHIGHSCHOOL) or (education==Education.HIGHSCHOOLGRADUATE):
            xb += np.log(1)
        elif education==Education.SOMECOLLEGE:
            xb += np.log(1.2)
        elif education==Education.COLLEGEGRADUATE:
            xb += np.log(0.8)
 
        if smokingStatus==SmokingStatus.FORMER:
            xb += np.log(1.2)
        elif smokingStatus==SmokingStatus.CURRENT:
            xb += np.log(2)
    
        if bmi<18.5:
            xb += np.log(1.2)
        elif 25<=bmi<30:
            xb += np.log(0.9)
        elif bmi>=20:
            xb += np.log(1.5)

        if (totChol>240) or (ldl>190):
            xb += np.log(1.2)

        if stroke==True:
            xb += np.log(2)
        if mi==True:
            xb += np.log(1.8)
        if diabetes==True:
            xb += np.log(1.5)
        if hypertension==True:
            xb += np.log(1.5)
        if gfr==True:
            xb += np.log(1.5)
            
        return xb


    def get_risk_for_person(self, person):
        lp = self.calc_linear_predictor_for_patient_characteristics(
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
            person._current_diabetes,
            person._any_antiHypertensive,
            person._gfr)

        return lp

    def generate_next_outcome(self, person):
        fatal = False
        return Outcome(OutcomeType.EPILEPSY, fatal)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person) if person._rng.uniform(size=1)<self.get_risk_for_person(person) else None
    