import numpy as np
import pandas as pd
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.person import Person
from microsim.outcome import OutcomeType
from collections import OrderedDict

# based on https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2805003, Model M2
class GCPStrokeModel:
    def __init__(self, outcomeModelRepository=None):
        self._outcome_model_repository = outcomeModelRepository
        pass

    def calc_linear_predictor_for_patient_characteristics(
        self,
        ageAtLastStroke,
        yearsSinceStroke,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        diabetes,
        physicalActivity,
        alcoholPerWeek,
        medianBmiPrestroke,
        meanSBP,
        meanSBPPrestroke,
        meanLdlPrestroke,
        meanLdl,
        gfr,
        medianWaistPrestroke,
        meanFastingGlucose,
        meanFastingGlucosePrestroke,
        anyAntiHypertensive,
        anyLipidLowering,
        afib,
        mi,
        medianGCPPrestroke):

       xb = 51.9602                                                       #Intercept
       xb += yearsSinceStroke * (-0.5249)                                 #slope, t_gcp_stk
       xb += (-1.4919) * (ageAtLastStroke/10.)                                #agemed10
       xb += (-0.1970) * (ageAtLastStroke/10.) * yearsSinceStroke             #t_gcp_stk*agemed10
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
       xb += 0.2301 * (meanSBP/10.)                                     #sbpmed10
       xb += 0.04248 * (meanSBP/10.) * yearsSinceStroke                 #sbpmed10*t_gcp_stk
       if anyAntiHypertensive:
           xb += (-1.3711)                                                #htntx
           xb += yearsSinceStroke * (0.2271)                              #t_gcp_stk*htntx
       xb += 0.1562 * (meanFastingGlucose/10.)                          #glucosefmed10
       xb += -0.04266 * (meanFastingGlucose/10.) * yearsSinceStroke     #t_gcp_stk*glucosefme
       xb += -0.02933 * meanFastingGlucosePrestroke                     #bs_glucosefmed10
       if afib:
           xb += -1.5329                                                  #Hxafib
       if mi:
           xb += 0.4470                                                   #HxMI
       if diabetes:                                                       #currently simulation can check for diabetes but there is no medication for that
           xb += -1.4601                                                  #diabetestx
           xb += (-0.03788) * yearsSinceStroke                            #t_gcp_stk*diabetestx
       xb += 0.01751 * gfr                                                #gfr
       xb += 0.6632 * medianGCPPrestroke                                  #bs_fgcpmed (I cannot tell exactly what this means, prestroke median gcp?
       xb += -0.2535 * meanSBPPrestroke                                 #bs_sbpstkcogmed10
       if anyLipidLowering:
           xb += -0.7570                                                  #choltx
           xb += 0.1035 * yearsSinceStroke                                #t_gcp_stk*choltx
       xb += -0.1866 * (meanLdlPrestroke/10.)                           #bs_cholldlmed10
       xb += -0.09122 * (meanLdl/10.)                                   #cholldlmed10
       xb += 0.007825 * (meanLdl/10.) * yearsSinceStroke                #t_gcp_stk*cholldlmed
       #weighted average to account for cohort (aric,chs,fos,regards-assumed to be baseline) 
       xb += (238.*(-5.2897)+332.*(-3.7359)+101.*(-2.8168)) / (238.+332.+101.+311.)           
       return xb       
                                                                  
    def get_risk_for_person(self, person, rng=None, years=1, vectorized=False, test=False):

        if not vectorized:
            random_effect = person._randomEffects["gcp"] if "gcp" in person._randomEffects else 0
        else:
            random_effect = rng.normal(0., 3.90) 

        residual = 0 if test else rng.normal(0, 6.08)

        linPred = 0
        if vectorized:
            ageAtLastStroke=person.ageAtLastStroke
            linPred = self.calc_linear_predictor_for_patient_characteristics(
                ageAtLastStroke=ageAtLastStroke,
                yearsSinceStroke=person.age-ageAtLastStroke,
                gender=person.gender,
                raceEthnicity=person.raceEthnicity,
                education=person.education,
                smokingStatus=person.smokingStatus,
                diabetes=person.current_diabetes,
                physicalActivity=person.anyPhysicalActivity,
                alcoholPerWeek=person.alcoholPerWeek,
                medianBmiPrestroke=person.medianBmiPriorToLastStroke,
                meanSBP=person.meanSbp,
                meanSBPPrestroke=person.meanSbpPriorToLastStroke,
                meanLdlPrestroke=person.meanLdlPriorToLastStroke,
                meanLdl=person.meanLdl,
                gfr=person.gfr,
                medianWaistPrestroke=person.medianWaistPriorToLastStroke,
                meanFastingGlucose=Person.convert_a1c_to_fasting_glucose(person.meanA1c),
                meanFastingGlucosePrestroke=Person.convert_a1c_to_fasting_glucose(person.meanA1cPriorToLastStroke),
                anyAntiHypertensive= ((person.antiHypertensiveCount + person.totalBPMedsAdded)> 0),
                anyLipidLowering= (person.statin | (person.otherLipidLoweringMedicationCount>0.)),
                afib=person.afib,
                mi=person.mi,
                medianGCPPrestroke=person.medianGcpPriorToLastStroke)
        else:
            ageAtLastStroke=person.get_age_at_last_outcome(OutcomeType.STROKE)
            waveAtLastStroke=person.get_wave_for_age(ageAtLastStroke)
            linPred = self.calc_linear_predictor_for_patient_characteristics(
                ageAtLastStroke=ageAtLastStroke,
                yearsSinceStroke=person._age[-1]-ageAtLastStroke,
                gender=person._gender,
                raceEthnicity=person._raceEthnicity,
                education=person._education,
                smokingStatus=person._smokingStatus,
                diabetes=person.has_diabetes(),
                physicalActivity=person._anyPhysicalActivity[-1],
                alcoholPerWeek=person._alcoholPerWeek[-1],
                medianBmiPrestroke=np.median(np.array(person._bmi[:waveAtLastStroke])),
                meanSBP=np.array(person._sbp).mean(),
                meanSBPPrestroke=np.array(person._sbp[:waveAtLastStroke]).mean(),
                meanLdlPrestroke=np.array(person._ldl[:waveAtLastStroke]).mean(),
                meanLdl=np.array(person._ldl).mean(),
                gfr=person._gfr,
                medianWaistPrestroke=np.median(np.array(person._waist[:waveAtLastStroke])),
                meanFastingGlucose=Person.convert_a1c_to_fasting_glucose(np.array(person._a1c).mean()),
                meanFastingGlucosePrestroke=Person.convert_a1c_to_fasting_glucose(np.array(person._a1c[:waveAtLastStroke]).mean()),
                anyAntiHypertensive=((person._antiHypertensiveCount[-1] + np.array(person._bpMedsAdded).sum()) > 0),
                anyLipidLowering= (person._statin[-1] | (person._otherLipidLoweringMedicationCount[-1]>0.)),
                afib=person._afib[-1],
                mi=person._mi,
                medianGCPPrestroke=np.median(np.array(person._gcp[:waveAtLastStroke])))    

        return linPred + random_effect + residual      
