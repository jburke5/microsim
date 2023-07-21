import numpy as np
import pandas as pd
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.person import Person
from collections import OrderedDict

# based on https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2805003, Model M2
class GCPStrokeModel:
    def __init__(self, outcomeModelRepository=None):
        self._outcome_model_repository = outcomeModelRepository
        pass

    def calc_linear_predictor_for_patient_characteristics(
        self,
        yearsSinceStroke,
        ageAtStroke,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        diabetes,
        physicalActivity,
        alcoholPerWeek,
        medianBmiPrestroke,
        medianSBP,
        medianSBPPrestroke,
        medianLdlPrestroke,
        medianLdl,
        gfr,
        medianWaistPrestroke,
        medianFastingGlucose,
        medianFastingGlucosePrestroke,
        anyAntiHypertensive,
        anyLipidLowering,
        afib,
        mi,
        medianGCPPrestroke):

       xb = 51.9602                                                       #Intercept
       xb += yearsSinceStroke * (-0.5249)                                 #slope, t_gcp_stk
       xb += (-1.4919) * (ageAtStroke/10.)                                #agemed10
       xb += (-0.1970) * (ageAtStroke/10.) * yearsSinceStroke             #t_gcp_stk*agemed10
       if gender == NHANESGender.FEMALE:
           xb += 1.4858                                                   #female0
           xb += yearsSinceStroke * (-0.2864)                             #t_gcp_stk*female0, change in the slope due to gender
       if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
           xb += -1.5739                                                  #black
       if education == Education.LESSTHANHIGHSCHOOL:
           xb += 0.                                                    
       elif education == Education.SOMEHIGHSCHOOL:
           xb += 0.                                                       
       elif education == Education.HIGHSCHOOLGRADUATE:
           xb += 0.9930                                                   #educ2
       elif education == Education.SOMECOLLEGE:    
           xb += -0.00267                                                 #educ3
       elif education == Education.COLLEGEGRADUATE:
           xb += 0.5712                                                   #educ4
       if smokingStatus == SmokingStatus.CURRENT:
           xb += 0.4707                                                   #currsmoker
       if physicalActivity:
           xb += 1.0047                                                   #physact
       xb += 0.05502 * alcoholPerWeek                                     #alcperwk
       xb += -0.1372 * medianBmiPrestroke                                 #bs_bmimed
       xb += 0.2726 * medianWaistPrestroke                                #bs_waistcmmed10
       xb += 0.2301 * (medianSBP/10.)                                     #sbpmed10
       xb += 0.04248 * (medianSBP/10.) * yearsSinceStroke                 #sbpmed10*t_gcp_stk
       if anyAntiHypertensive:
           xb += (-1.3711)                                                #htntx
           xb += yearsSinceStroke * (0.2271)                              #t_gcp_stk*htntx
       xb += 0.1562 * (medianFastingGlucose/10.)                          #glucosefmed10
       xb += -0.04266 * (medianFastingGlucose/10.) * yearsSinceStroke     #t_gcp_stk*glucosefme
       xb += -0.02933 * medianFastingGlucosePrestroke                     #bs_glucosefmed10
       if afib:
           xb += -1.5329                                                  #Hxafib
       if mi:
           xb += 0.4470                                                   #HxMI
       if diabetes:                                                       #currently simulation can check for diabetes but there is no medication for that
           xb += -1.4601                                                  #diabetestx
           xb += (-0.03788) * yearsSinceStroke                            #t_gcp_stk*diabetestx
       xb += 0.01751 * gfr                                                #gfr
       xb += 0.6632 * medianGCPPrestroke                                  #bs_fgcpmed (I cannot tell exactly what this means, prestroke median gcp?
                                                                          #aric, chs, fos (not implemented in sim)
       xb += -0.2535 * medianSBPPrestroke                                 #bs_sbpstkcogmed10
       if anyLipidLowering:
           xb += -0.7570                                                  #choltx
           xb += 0.1035 * yearsSinceStroke                                #t_gcp_stk*choltx
       xb += -0.1866 * (medianLdlPrestroke/10.)                           #bs_cholldlmed10
       xb += -0.09122 * (medianLdl/10.)                                   #cholldlmed10
       xb += 0.007825 * (medianLdl/10.) * yearsSinceStroke                #t_gcp_stk*cholldlmed
       return xb       
                                                                  
    def get_risk_for_person(self, person, rng=None, years=1, vectorized=False, test=False):
        return 50.
        random_effect = rng.normal(0., 3.90) 

        residual = 0 if test else rng.normal(0, 6.08)

        linPred = 0
       # if vectorized:
           # linPred = self.calc_linear_predictor_for_patient_characteristics(
            #    yearsSinceStroke,
             #   ageAtStroke,
              #  gender=person.gender,
               # raceEthnicity=person.raceEthnicity,
               # education=person.education,
               # smokingStatus=person.smokingStatus,
               # diabetes,
               # physicalActivity=person.anyPhysicalActivity,
               # alcoholPerWeek=person.alcoholPerWeek,
               # medianBmiPrestroke,
               # medianSBP,
               # medianSBPPrestroke,
               # medianLdlPrestroke,
               # medianLdl,
               # gfr,
               # medianWaistPrestroke,
               # medianFastingGlucose,
               # medianFastingGlucosePrestroke,
               # anyAntiHypertensive=((person.antiHypertensiveCount + person.totalBPMedsAdded)> 0),
               # anyLipidLowering,
               # afib=person.afib,
               # mi,
               # medianGCPPrestroke)
        #else:
            #not implemented yet

        #return linPred + random_effect + residual      
