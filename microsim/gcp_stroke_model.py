import numpy as np
import pandas as pd
from microsim.smoking_status import SmokingStatus
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.person import Person
from microsim.outcome import OutcomeType
from microsim.alcohol_category import AlcoholCategory
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
        #diabetestx, #simulation does not currently include diabetes treatment
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

       #standardize some variables first, if a variable is ending on "med10" that meant it was centered and standardized by 10
       ageAtLastStrokeS = (ageAtLastStroke-74.6)/10.
       medianBmiPrestrokeS = (medianBmiPrestroke-27.2)
       medianWaistPrestrokeS = (medianWaistPrestroke-97.5)/10.
       meanSBPS = (meanSBP-134.9)/10.
       meanFastingGlucoseS = (meanFastingGlucose-108.1)/10.
       meanFastingGlucosePrestrokeS = (meanFastingGlucosePrestroke-112.8)/10.
       medianGCPPrestrokeS = medianGCPPrestroke - 52.7 
       meanSBPPrestrokeS = (meanSBPPrestroke-140.4)/10.
       meanLdlS = (meanLdl-94.1)/10.
       meanLdlPrestrokeS = (meanLdlPrestroke - 126.4)/10.

       xb = 51.9602                                                #Intercept
       xb += yearsSinceStroke * (-0.5249)                          #slope, t_gcp_stk
       xb += (-1.4919) * ageAtLastStrokeS                          #agemed10
       xb += (-0.1970) * ageAtLastStrokeS * yearsSinceStroke       #t_gcp_stk*agemed10
       if gender == NHANESGender.FEMALE:
           xb += 1.4858                                            #female0
           xb += yearsSinceStroke * (-0.2864)                      #t_gcp_stk*female0, change in the slope due to gender
       if raceEthnicity == NHANESRaceEthnicity.NON_HISPANIC_BLACK:
           xb += -1.5739                                           #black
       if education == Education.HIGHSCHOOLGRADUATE:
           xb += 0.9930                                            #educ2
       elif education == Education.SOMECOLLEGE:    
           xb += -0.00267                                          #educ3
       elif education == Education.COLLEGEGRADUATE:
           xb += 0.5712                                            #educ4
       if smokingStatus == SmokingStatus.CURRENT:
           xb += 0.4707                                            #currsmoker
       if physicalActivity:
           xb += 1.0047                                            #physact
       if alcoholPerWeek == AlcoholCategory.ONETOSIX:              #alcperwk
           xb += 0.05502 * 3.5                                     
       elif alcoholPerWeek == AlcoholCategory.SEVENTOTHIRTEEN:
           xb += 0.05502 * 10
       elif alcoholPerWeek == AlcoholCategory.FOURTEENORMORE:
           xb += 0.05502 * 17
       xb += -0.1372 * medianBmiPrestrokeS                         #bs_bmimed
       xb += 0.2726 * medianWaistPrestrokeS                        #bs_waistcmmed10
       xb += 0.2301 * meanSBPS                                     #sbpmed10
       xb += 0.04248 * meanSBPS * yearsSinceStroke                 #sbpmed10*t_gcp_stk
       if anyAntiHypertensive:
           xb += (-1.3711)                                         #htntx
           xb += yearsSinceStroke * (0.2271)                       #t_gcp_stk*htntx
       xb += 0.1562 * meanFastingGlucoseS                          #glucosefmed10
       xb += -0.04266 * meanFastingGlucoseS * yearsSinceStroke     #t_gcp_stk*glucosefme
       xb += -0.02933 * meanFastingGlucosePrestrokeS               #bs_glucosefmed10
       if afib:
           xb += -1.5329                                           #Hxafib
       if mi:
           xb += 0.4470                                            #HxMI
       #if diabetestx:                                                #currently simulation does not include diabetes medication
       #    xb += -1.4601                                           #diabetestx
       #    xb += (-0.03788) * yearsSinceStroke                     #t_gcp_stk*diabetestx
       xb += 0.01751 * gfr                                         #gfr
       xb += 0.6632 * medianGCPPrestrokeS                          #bs_fgcpmed (I cannot tell exactly what this means, prestroke median gcp?
       xb += -0.2535 * meanSBPPrestrokeS                           #bs_sbpstkcogmed10
       if anyLipidLowering:
           xb += -0.7570                                           #choltx
           xb += 0.1035 * yearsSinceStroke                         #t_gcp_stk*choltx
       xb += -0.1866 * meanLdlPrestrokeS                           #bs_cholldlmed10
       xb += -0.09122 * meanLdlS                                   #cholldlmed10
       xb += 0.007825 * meanLdlS * yearsSinceStroke                #t_gcp_stk*cholldlmed
       #weighted average to account for cohort (aric,chs,fos,regards-assumed to be baseline) 
       xb += (238.*(-5.2897)+332.*(-3.7359)+101.*(-2.8168)) / (238.+332.+101.+311.)           
       return xb       
                                                                  
    def get_risk_for_person(self, person, rng=None, years=1, vectorized=False, test=False):

        if not vectorized:
            random_effect = person._randomEffects["gcp"] if "gcp" in person._randomEffects else 0
        else:
            random_effect = 0 if test else rng.normal(0., 3.90) 

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
                #diabetes=person.current_diabetestx,
                physicalActivity=person.anyPhysicalActivity,
                alcoholPerWeek=person.alcoholPerWeek,
                medianBmiPrestroke=person.medianBmiPriorToLastStroke,
                meanSBP=person.meanSbpSinceLastStroke,
                meanSBPPrestroke=person.meanSbpPriorToLastStroke,
                meanLdlPrestroke=person.meanLdlPriorToLastStroke,
                meanLdl=person.meanLdlSinceLastStroke,
                gfr=person.gfr,
                medianWaistPrestroke=person.medianWaistPriorToLastStroke,
                meanFastingGlucose=Person.convert_a1c_to_fasting_glucose(person.meanA1cSinceLastStroke),
                meanFastingGlucosePrestroke=Person.convert_a1c_to_fasting_glucose(person.meanA1cPriorToLastStroke),
                anyAntiHypertensive= ((person.antiHypertensiveCount + person.totalBPMedsAdded)> 0),
                anyLipidLowering= (person.statin | (person.otherLipidLoweringMedicationCount>0.)),
                afib=person.afib,
                mi=person.mi,
                medianGCPPrestroke=person.medianGcpPriorToLastStroke)
        else:
            ageAtLastStroke=person.get_age_at_last_outcome(OutcomeType.STROKE)
            #the get_wave function gives me the wave that follows the updates to the person object, but I want the wave when the last updates took place
            waveAtLastStroke=person.get_wave_for_age(ageAtLastStroke)-1
            linPred = self.calc_linear_predictor_for_patient_characteristics(
                ageAtLastStroke=ageAtLastStroke,
                yearsSinceStroke=person._age[-1]-ageAtLastStroke,
                gender=person._gender,
                raceEthnicity=person._raceEthnicity,
                education=person._education,
                smokingStatus=person._smokingStatus,
                #diabetes=person.has_diabetestx(),
                physicalActivity=person._anyPhysicalActivity[-1],
                alcoholPerWeek=person._alcoholPerWeek[-1],
                medianBmiPrestroke=np.median(np.array(person._bmi[:waveAtLastStroke+1])),
                meanSBP=np.array(person._sbp[waveAtLastStroke+1:]).mean(),
                meanSBPPrestroke=np.array(person._sbp[:waveAtLastStroke+1]).mean(),
                meanLdlPrestroke=np.array(person._ldl[:waveAtLastStroke+1]).mean(),
                meanLdl=np.array(person._ldl[waveAtLastStroke+1:]).mean(),
                gfr=person._gfr,
                medianWaistPrestroke=np.median(np.array(person._waist[:waveAtLastStroke+1])),
                meanFastingGlucose=Person.convert_a1c_to_fasting_glucose(np.array(person._a1c[waveAtLastStroke+1:]).mean()),
                meanFastingGlucosePrestroke=Person.convert_a1c_to_fasting_glucose(np.array(person._a1c[:waveAtLastStroke+1]).mean()),
                anyAntiHypertensive=person._current_bp_treatment,
                anyLipidLowering= (person._statin[-1] | (person._otherLipidLoweringMedicationCount[-1]>0.)),
                afib=person._afib[-1],
                mi=person._mi,
                medianGCPPrestroke=np.median(np.array(person._gcp[:waveAtLastStroke+1])))    

        return linPred + random_effect + residual      
