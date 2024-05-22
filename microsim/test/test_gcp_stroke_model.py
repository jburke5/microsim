import unittest
import numpy as np
import pandas as pd

from microsim.person import Person
from microsim.population import Population
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.alcohol_category import AlcoholCategory
from microsim.race_ethnicity import NHANESRaceEthnicity
from microsim.outcome import Outcome, OutcomeType
from microsim.gcp_stroke_model import GCPStrokeModel
from microsim.population_factory import PopulationFactory
from microsim.risk_factor import StaticRiskFactorsType, DynamicRiskFactorsType
from microsim.treatment import DefaultTreatmentsType
from microsim.person_factory import PersonFactory
from microsim.cognition_outcome import CognitionOutcome

#main idea: construct persons that at some point in their simulation history had a stroke outcome
#we want to test the GCP stroke model with persons that resemble simulation persons as much as possible
#for every person, lists are created, these lists define the person's history in the simulation
#these lists have the form [prestroke value, ..., prestroke value, after stroke value, ... after stroke value]
#with the stroke outcome taking place at the wave of the last prestroke value
#prestroke values and after stroke values were taken from data used to develop this model
#for variables where only prestroke values contributed to the model the lists had the form 
# [prestroke value, ..., prestroke value, prestroke value + a number, prestroke value + another number...]
#so that we could see if any after stroke value was accidentally included in our implementation of the model
#also, the current GCP stroke model implementation does not take every model factor into account so at the end for each
#test case presented here we need to adjust for the cohort (remove the weighted average our implementation includes and add
#the correct cohort), income, diabetes treatment (2 terms), random effects (2 terms), remove the average alcohol per week
#term our implementation includes and add the correct one
#tried to include test cases with diverse histories so that we can test as many model components as possible (I think no test case had afib though)

initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

#row 2 in excel file 
class TestCaseOne(Person):

    def __init__(self):

        #make the lists that will be used to define the person's path in the simulation
        ageAtStroke = 65 + 0.389117043*10                        
        indexStroke = 2
        ageList = [ageAtStroke + 3.17 + x for x in range(-5,1)]
        ageList.insert(indexStroke, ageAtStroke)
        sbpMeanPrestroke = 130. - 1.9*10
        sbpMean = -1.*10 + 130.
        sbpList = [sbpMeanPrestroke] * (indexStroke+1) + [sbpMean]*4
        dbpList = [80]*7
        a1cMeanPrestroke = Person.convert_fasting_glucose_to_a1c(100. + 0.5*10)
        a1cMean = Person.convert_fasting_glucose_to_a1c(100. + 2.2*10)
        a1cList = [a1cMeanPrestroke] * (indexStroke+1) + [a1cMean]*4
        hdlList = [50]*7
        ldlMeanPrestroke = 93. + 3.8*10
        ldlMean = 2.6*10 + 93.
        ldlList = [ldlMeanPrestroke] * (indexStroke+1) + [ldlMean]*4
        trigList = [150]*7
        totCholList = [hdlList[i]+ldlList[i]+0.2*trigList[i] for i in range(len(ldlList))]
        bmiMeanPrestroke = 25. + 1.467401286
        bmiList = [bmiMeanPrestroke] * (indexStroke+1) + [bmiMeanPrestroke+x for x in range(10,50,10)]
        waistMeanPrestroke = 100 - 1.618*10
        waistList = [waistMeanPrestroke] * (indexStroke+1) + [waistMeanPrestroke+x for x in range(10,50,10)]
        antiHypertensiveCountList = [0]*7
        statinList = [1]*7
        otherLipidLoweringMedicationCountList = [0]*7
        creatinineList = [1]*7
        anyPhysicalActivityList=[1]*7
        meanGcpPrestroke = 6.104780222+50.
        gcpList = [meanGcpPrestroke]*7
    
        x = pd.DataFrame({DynamicRiskFactorsType.AGE.value: ageList[0],    #agemed10
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,  #female0
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_WHITE.value,  #black
                               DynamicRiskFactorsType.SBP.value: sbpList[0],   #bs_sbpstkcog
                               DynamicRiskFactorsType.DBP.value: dbpList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.A1C.value: a1cList[0],   #bs_glucosefmed10
                               DynamicRiskFactorsType.HDL.value: hdlList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.TOT_CHOL.value: totCholList[0],  #?? hdl+ldl+0.2*trig
                               DynamicRiskFactorsType.BMI.value: bmiList[0],    #bs_bmimed
                               DynamicRiskFactorsType.LDL.value: ldlList[0],    #bs_cholldlmed10
                               DynamicRiskFactorsType.TRIG.value: trigList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.WAIST.value: waistList[0],  #bs_waistcmmed10
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: anyPhysicalActivityList[0],  #physact
                               StaticRiskFactorsType.EDUCATION.value: Education.SOMEHIGHSCHOOL.value,   #educ2,educ3,educ4
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,   #currsmoker
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,  #alcperwk
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: antiHypertensiveCountList[0],  #htntx
                               DefaultTreatmentsType.STATIN.value: statinList[0],  #choltx
                               DynamicRiskFactorsType.CREATININE.value: creatinineList[0], #same as TestGCPModel
                               "name": "one"}, index=[0])
        (name,
         personStaticRiskFactors,
         personDynamicRiskFactors,
         personDefaultTreatments,
         personTreatmentStrategies,
         personOutcomes) =  PersonFactory.get_nhanes_person_init_information(x.iloc[0])

        #create the person
        super().__init__(name,
         personStaticRiskFactors,
         personDynamicRiskFactors,
         personDefaultTreatments,
         personTreatmentStrategies,
         personOutcomes)
        self._afib = [False]
        
        #assign history to person
        self.add_outcome(CognitionOutcome(False, False, gcpList[0]))
        for i,age in enumerate(ageList[1:indexStroke+1]):
            self._age.append(ageList[i+1])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+1]))
        #when age reaches ageAtStroke, then add the stroke outcome
        self.add_outcome(Outcome(OutcomeType.STROKE, False))    
        for ii, age in enumerate(ageList[indexStroke+1:]):
            self._age.append(ageList[i+ii+2])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+ii+2]))
        self._sbp = sbpList
        self._dbp = dbpList
        self._a1c = a1cList
        self._hdl = hdlList
        self._totChol = totCholList
        self._ldl = ldlList
        self._trig = trigList
        self._bmi = bmiList
        self._waist = waistList
        self._anyPhysicalActivity = anyPhysicalActivityList
        self._antiHypertensiveCount = antiHypertensiveCountList
        self._statin = statinList
        self._creatinine = creatinineList
        self._randomEffects={"gcp": 0,
                             "gcpStroke": 1.989499,
                             "gcpStrokeSlope": 0.0925494}

        #expected results based on model
        self._expectedGcp = 58.45580379
        self._expectedYhat = 57.4753
        self._expectedResidual = 0.9805041
        self._expectedLinearPredictor = (self._expectedYhat +
                                         0.05502 * 0 -          #the alcperwk contribution in microsim
                                         0.05502 * 0. +         #the alcperwk contribution on the model
                                         (238.*(-5.2897)+332.*(-3.7359)+101.*(-2.8168)) / (238.+332.+101.+311.) -  #weighted group average
                                         0. -                   #group
                                         0.4819 -               #income
                                         0. -                   #diabetes treatment
                                         0.*(-0.03788) )        #diabetes treatment * t_gcp_stk
    
    @property
    def _gfr(self):
        return 80.61366 #override the gfr calculation and use the actual measurement 

#row 29 in excel file 
class TestCaseTwo(Person):

    def __init__(self):

        #make the lists that will be used to define the person's path in the simulation
        ageAtStroke = 65 + 1.81013*10
        indexStroke = 2
        ageList = [ageAtStroke + 5.43189 + x for x in range(-7,1)]
        ageList.insert(indexStroke, ageAtStroke)
        sbpMeanPrestroke = 130. + 1.6*10
        sbpMean = 130. - 0.8*10 
        sbpList = [sbpMeanPrestroke] * (indexStroke+1) + [sbpMean]*6
        dbpList = [80]*9
        a1cMeanPrestroke = Person.convert_fasting_glucose_to_a1c(100. - 0.8*10)
        a1cMean = Person.convert_fasting_glucose_to_a1c(100. - 0.1*10)
        a1cList = [a1cMeanPrestroke] * (indexStroke+1) + [a1cMean]*6
        hdlList = [50]*9
        ldlMeanPrestroke = 93. + 3.8*10
        ldlMean = 93. + 0.21*10 
        ldlList = [ldlMeanPrestroke] * (indexStroke+1) + [ldlMean]*6
        trigList = [150]*9
        totCholList = [hdlList[i]+ldlList[i]+0.2*trigList[i] for i in range(len(ldlList))]
        bmiMeanPrestroke = 25. + -2.36183
        bmiList = [bmiMeanPrestroke] * (indexStroke+1) + [bmiMeanPrestroke+x for x in range(10,70,10)]
        waistMeanPrestroke = 100. - 1.237*10
        waistList = [waistMeanPrestroke] * (indexStroke+1) + [waistMeanPrestroke+x for x in range(10,70,10)]
        antiHypertensiveCountList = [0]*9
        statinList = [1]*9
        otherLipidLoweringMedicationCountList = [0]*9
        creatinineList = [1]*9
        anyPhysicalActivityList=[1]*9
        meanGcpPrestroke = 50. + 5.40984283
        gcpList = [meanGcpPrestroke]*9

        x = pd.DataFrame({DynamicRiskFactorsType.AGE.value: ageList[0],    #agemed10
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,  #female0
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_BLACK.value,  #black
                               DynamicRiskFactorsType.SBP.value: sbpList[0],   #bs_sbpstkcog
                               DynamicRiskFactorsType.DBP.value: dbpList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.A1C.value: a1cList[0],   #bs_glucosefmed10
                               DynamicRiskFactorsType.HDL.value: hdlList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.TOT_CHOL.value: totCholList[0],  #?? hdl+ldl+0.2*trig
                               DynamicRiskFactorsType.BMI.value: bmiList[0],    #bs_bmimed
                               DynamicRiskFactorsType.LDL.value: ldlList[0],    #bs_cholldlmed10
                               DynamicRiskFactorsType.TRIG.value: trigList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.WAIST.value: waistList[0],  #bs_waistcmmed10
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: anyPhysicalActivityList[0],  #physact
                               StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,   #educ2,educ3,educ4
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,   #currsmoker
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.ONETOSIX.value,  #alcperwk
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: antiHypertensiveCountList[0],  #htntx
                               DefaultTreatmentsType.STATIN.value: statinList[0],  #choltx
                               DynamicRiskFactorsType.CREATININE.value: creatinineList[0], #same as TestGCPModel
                               "name": "two"}, index=[0])
        (name,
         personStaticRiskFactors,
         personDynamicRiskFactors,
         personDefaultTreatments,
         personTreatmentStrategies,
         personOutcomes) =  PersonFactory.get_nhanes_person_init_information(x.iloc[0])

        #create the person
        super().__init__(name,
         personStaticRiskFactors,
         personDynamicRiskFactors,
         personDefaultTreatments,
         personTreatmentStrategies,
         personOutcomes)
        self._afib = [False]

        #assign history to person
        self.add_outcome(CognitionOutcome(False, False, gcpList[0]))
        for i,age in enumerate(ageList[1:indexStroke+1]):
            self._age.append(ageList[i+1])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+1]))
        #when age reaches ageAtStroke, then add the stroke outcome
        self.add_outcome(Outcome(OutcomeType.STROKE, False))    
        for ii, age in enumerate(ageList[indexStroke+1:]):
            self._age.append(ageList[i+ii+2])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+ii+2])) 
        self._sbp = sbpList
        self._dbp = dbpList
        self._a1c = a1cList
        self._hdl = hdlList
        self._totChol = totCholList                                   
        self._ldl = ldlList
        self._trig = trigList
        self._bmi = bmiList
        self._waist = waistList
        self._anyPhysicalActivity = anyPhysicalActivityList               
        self._antiHypertensiveCount = antiHypertensiveCountList       
        self._statin = statinList                                     
        self._creatinine = creatinineList                             
        self._randomEffects={"gcp": 0,
                             "gcpStroke": 2.216342,
                             "gcpStrokeSlope": 0.0920485}         
  
        #expected results based on model
        self._expectedGcp = 45.043121
        self._expectedYhat = 50.527
        self._expectedResidual = -5.483882
        self._expectedLinearPredictor = (self._expectedYhat +
                                         0.05502 * 3.5 -          #the alcperwk contribution in microsim
                                         0.05502 * 1. +         #the alcperwk contribution on the model
                                         (238.*(-5.2897)+332.*(-3.7359)+101.*(-2.8168)) / (238.+332.+101.+311.) -  #weighted group average
                                         0. -                   #group
                                         0.05448 -               #income
                                         0. -                   #diabetes treatment
                                         0.*(-0.03788) )        #diabetes treatment * t_gcp_stk
            
    @property
    def _gfr(self):
        return 74.86089 #override the gfr calculation and use the actual measurement 

#row 18 in excel file 
class TestCaseThree(Person):

    def __init__(self):

        #make the lists that will be used to define the person's path in the simulation
        ageAtStroke = 65 + -1.21225*10
        indexStroke = 2
        ageList = [ageAtStroke + 3.3976728 + x for x in range(-5,1)]
        ageList.insert(indexStroke, ageAtStroke)
        sbpMeanPrestroke = 130. + 0.93285*10
        sbpMean = 130. - 0.1*10
        sbpList = [sbpMeanPrestroke] * (indexStroke+1) + [sbpMean]*4
        dbpList = [80]*7
        a1cMeanPrestroke = Person.convert_fasting_glucose_to_a1c(100. + 1.1*10)
        a1cMean = Person.convert_fasting_glucose_to_a1c(100. + 1.0*10)
        a1cList = [a1cMeanPrestroke] * (indexStroke+1) + [a1cMean]*4
        hdlList = [50]*7
        ldlMeanPrestroke = 93. + 1.6571428*10
        ldlMean = 93. + -0.2*10
        ldlList = [ldlMeanPrestroke] * (indexStroke+1) + [ldlMean]*4
        trigList = [150]*7
        totCholList = [hdlList[i]+ldlList[i]+0.2*trigList[i] for i in range(len(ldlList))]
        bmiMeanPrestroke = 25. + 2.7184726
        bmiList = [bmiMeanPrestroke] * (indexStroke+1) + [bmiMeanPrestroke+x for x in range(10,50,10)]
        waistMeanPrestroke = 100. - 0.475*10
        waistList = [waistMeanPrestroke] * (indexStroke+1) + [waistMeanPrestroke+x for x in range(10,50,10)]
        antiHypertensiveCountList = [1]*7
        statinList = [0]*7
        otherLipidLoweringMedicationCountList = [0]*7
        creatinineList = [1]*7
        anyPhysicalActivityList=[1]*7
        meanGcpPrestroke = 50. + 13.656746
        gcpList = [meanGcpPrestroke]*7

        #create the person
        x = pd.DataFrame({DynamicRiskFactorsType.AGE.value: ageList[0],    #agemed10
                               StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,  #female0
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_WHITE.value,  #black
                               DynamicRiskFactorsType.SBP.value: sbpList[0],   #bs_sbpstkcog
                               DynamicRiskFactorsType.DBP.value: dbpList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.A1C.value: a1cList[0],   #bs_glucosefmed10
                               DynamicRiskFactorsType.HDL.value: hdlList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.TOT_CHOL.value: totCholList[0],  #?? hdl+ldl+0.2*trig
                               DynamicRiskFactorsType.BMI.value: bmiList[0],    #bs_bmimed
                               DynamicRiskFactorsType.LDL.value: ldlList[0],    #bs_cholldlmed10
                               DynamicRiskFactorsType.TRIG.value: trigList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.WAIST.value: waistList[0],  #bs_waistcmmed10
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: anyPhysicalActivityList[0],  #physact
                               StaticRiskFactorsType.EDUCATION.value: Education.SOMECOLLEGE.value,   #educ2,educ3,educ4
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.CURRENT.value,   #currsmoker
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.FOURTEENORMORE.value,  #alcperwk
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: antiHypertensiveCountList[0],  #htntx
                               DefaultTreatmentsType.STATIN.value: statinList[0],  #choltx
                               DynamicRiskFactorsType.CREATININE.value: creatinineList[0], #same as TestGCPModel
                               "name": "three"}, index=[0])

        (name,
         personStaticRiskFactors,
         personDynamicRiskFactors,
         personDefaultTreatments,
         personTreatmentStrategies,
         personOutcomes) =  PersonFactory.get_nhanes_person_init_information(x.iloc[0])

        #create the person
        super().__init__(name,
         personStaticRiskFactors,
         personDynamicRiskFactors,
         personDefaultTreatments,
         personTreatmentStrategies,
         personOutcomes)
        self._afib = [False]

        #assign history to person
        self.add_outcome(CognitionOutcome(False, False, gcpList[0]))
        for i,age in enumerate(ageList[1:indexStroke+1]):
            self._age.append(ageList[i+1])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+1]))
        #when age reaches ageAtStroke, then add the stroke outcome
        self.add_outcome(Outcome(OutcomeType.STROKE, False))
        for ii, age in enumerate(ageList[indexStroke+1:]):
            self._age.append(ageList[i+ii+2])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+ii+2]))
        self._sbp = sbpList
        self._dbp = dbpList
        self._a1c = a1cList
        self._hdl = hdlList
        self._totChol = totCholList                                   
        self._ldl = ldlList
        self._trig = trigList
        self._bmi = bmiList
        self._waist = waistList
        self._anyPhysicalActivity = anyPhysicalActivityList               
        self._antiHypertensiveCount = antiHypertensiveCountList       
        self._statin = statinList                                     
        self._creatinine = creatinineList                             
        self._randomEffects={"gcp": 0,
                             "gcpStroke": 0.5671927,                   
                             "gcpStrokeSlope": 0.1222775}   
           
        #expected results based on model
        self._expectedGcp = 52.8837385
        self._expectedYhat = 62.5652
        self._expectedResidual = -9.681457
        self._expectedLinearPredictor = (self._expectedYhat +
                                         0.05502 * 17. -         #the alcperwk contribution in microsim
                                         0.05502 * 24. +         #the alcperwk contribution on the model
                                         (238.*(-5.2897)+332.*(-3.7359)+101.*(-2.8168)) / (238.+332.+101.+311.) -  #weighted group average
                                         -2.8168 -                   #group
                                         0.05448 -               #income
                                         0. -                   #diabetes treatment
                                         0.*(-0.03788) )        #diabetes treatment * t_gcp_stk

    @property
    def _gfr(self):
        return 50.147722 #override the gfr calculation and use the actual measurement 

#row 38 in excel file 
class TestCaseFour(Person):
        
    def __init__(self):
        
        #make the lists that will be used to define the person's path in the simulation
        ageAtStroke = 65. + 0.85400*10
        indexStroke = 2
        ageList = [ageAtStroke + 8.265571 + x for x in range(-10,1)]
        ageList.insert(indexStroke, ageAtStroke) 
        sbpMeanPrestroke = 130. + 1.238571*10
        sbpMean = 130. + 2.0333*10         
        sbpList = [sbpMeanPrestroke] * (indexStroke+1) + [sbpMean]*9
        dbpList = [80]*12                 
        a1cMeanPrestroke = Person.convert_fasting_glucose_to_a1c(100. + - 0.030749*10)
        a1cMean = Person.convert_fasting_glucose_to_a1c(100. + -0.252000*10)
        a1cList = [a1cMeanPrestroke] * (indexStroke+1) + [a1cMean]*9
        hdlList = [50]*12                 
        ldlMeanPrestroke = 93. + 6.2865*10
        ldlMean = 93. + -1.2*10
        ldlList = [ldlMeanPrestroke] * (indexStroke+1) + [ldlMean]*9
        trigList = [150]*12
        totCholList = [hdlList[i]+ldlList[i]+0.2*trigList[i] for i in range(len(ldlList))]
        bmiMeanPrestroke = 25. + 2.295926
        bmiList = [bmiMeanPrestroke] * (indexStroke+1) + [bmiMeanPrestroke+x for x in range(10,100,10)]
        waistMeanPrestroke = 100. + 0.03333*10
        waistList = [waistMeanPrestroke] * (indexStroke+1) + [waistMeanPrestroke+x for x in range(10,100,10)]
        antiHypertensiveCountList = [1]*12
        statinList = [1]*12
        otherLipidLoweringMedicationCountList = [0]*12
        creatinineList = [1]*12
        anyPhysicalActivityList=[1]*12
        meanGcpPrestroke = 50. + 2.946463
        gcpList = [meanGcpPrestroke]*12

        #create the person
        x = pd.DataFrame({DynamicRiskFactorsType.AGE.value: ageList[0],    #agemed10
                               StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,  #female0
                               StaticRiskFactorsType.RACE_ETHNICITY.value:NHANESRaceEthnicity.NON_HISPANIC_WHITE.value,  #black
                               DynamicRiskFactorsType.SBP.value: sbpList[0],   #bs_sbpstkcog
                               DynamicRiskFactorsType.DBP.value: dbpList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.A1C.value: a1cList[0],   #bs_glucosefmed10
                               DynamicRiskFactorsType.HDL.value: hdlList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.TOT_CHOL.value: totCholList[0],  #?? hdl+ldl+0.2*trig
                               DynamicRiskFactorsType.BMI.value: bmiList[0],    #bs_bmimed
                               DynamicRiskFactorsType.LDL.value: ldlList[0],    #bs_cholldlmed10
                               DynamicRiskFactorsType.TRIG.value: trigList[0],   #same as TestGCPModel
                               DynamicRiskFactorsType.WAIST.value: waistList[0],  #bs_waistcmmed10
                               DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: anyPhysicalActivityList[0],  #physact
                               StaticRiskFactorsType.EDUCATION.value: Education.HIGHSCHOOLGRADUATE.value,   #educ2,educ3,educ4
                               StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,   #currsmoker
                               DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholCategory.NONE.value,  #alcperwk
                               DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: antiHypertensiveCountList[0],  #htntx
                               DefaultTreatmentsType.STATIN.value: statinList[0],  #choltx
                               DynamicRiskFactorsType.CREATININE.value: creatinineList[0], #same as TestGCPModel
                               "name": "four"}, index=[0])
        (name,
         personStaticRiskFactors,
         personDynamicRiskFactors,
         personDefaultTreatments,
         personTreatmentStrategies,
         personOutcomes) =  PersonFactory.get_nhanes_person_init_information(x.iloc[0])
        
        #create the person 
        super().__init__(name,
         personStaticRiskFactors, 
         personDynamicRiskFactors,
         personDefaultTreatments, 
         personTreatmentStrategies,
         personOutcomes)
        self._afib = [False]
        
        #assign history to person
        self.add_outcome(CognitionOutcome(False, False, gcpList[0]))
        for i,age in enumerate(ageList[1:indexStroke+1]):
            self._age.append(ageList[i+1])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+1])) 
        #when age reaches ageAtStroke, then add the stroke outcome 
        self.add_outcome(Outcome(OutcomeType.STROKE, False))
        for ii, age in enumerate(ageList[indexStroke+1:]):
            self._age.append(ageList[i+ii+2])
            self.add_outcome(CognitionOutcome(False, False, gcpList[i+ii+2]))  
        self._sbp = sbpList
        self._dbp = dbpList                                           
        self._a1c = a1cList
        self._hdl = hdlList
        self._totChol = totCholList                                   
        self._ldl = ldlList
        self._trig = trigList                                         
        self._bmi = bmiList
        self._waist = waistList
        self._anyPhysicalActivity = anyPhysicalActivityList               
        self._antiHypertensiveCount = antiHypertensiveCountList         
        self._statin = statinList                                               
        self._creatinine = creatinineList                             
        self._randomEffects={"gcp": 0,
                             "gcpStroke": 1.410774,
                             "gcpStrokeSlope": 0.0976629}
            
        #expected results based on model                              
        self._expectedGcp = 50.834723
        self._expectedYhat = 49.26757
        self._expectedResidual = 1.567155
        self._expectedLinearPredictor = (self._expectedYhat +
                                         0.05502 * 0. -         #the alcperwk contribution in microsim
                                         0.05502 * 0.5 +         #the alcperwk contribution on the model
                                         (238.*(-5.2897)+332.*(-3.7359)+101.*(-2.8168)) / (238.+332.+101.+311.) -  #weighted group average
                                         -3.7359 -                   #group
                                         0.05448 -               #income
                                         0. -                   #diabetes treatment
                                         0.*(-0.03788) )        #diabetes treatment * t_gcp_stk

    @property
    def _gfr(self):
        return 72.66258 #override the gfr calculation and use the actual measurement 

class TestGCPStrokeModel(unittest.TestCase):

    def setUp(self):

        self._test_case_one = TestCaseOne()
        self._test_case_two = TestCaseTwo()
        self._test_case_three = TestCaseThree()
        self._test_case_four = TestCaseFour()

    def test_expected_linear_predictor(self):

        self.assertAlmostEqual(
            GCPStrokeModel().get_risk_for_person(self._test_case_one, rng=None, years=1, vectorized=False, test=True),
            self._test_case_one._expectedLinearPredictor,
            places=2)
            
        self.assertAlmostEqual(
            GCPStrokeModel().get_risk_for_person(self._test_case_two, rng=None, years=1, vectorized=False, test=True),
            self._test_case_two._expectedLinearPredictor,
            places=2)

        self.assertAlmostEqual(
            GCPStrokeModel().get_risk_for_person(self._test_case_three, rng=None, years=1, vectorized=False, test=True),
            self._test_case_three._expectedLinearPredictor,
            places=2)

        self.assertAlmostEqual(
            GCPStrokeModel().get_risk_for_person(self._test_case_four, rng=self._test_case_four._rng, years=1, vectorized=False, test=True),
            self._test_case_four._expectedLinearPredictor,
            places=2)

    #with the random effects now part of get_risk_for_person method I think this test is no longer possible
    def test_random_effect(self):
    
         self._test_case_one._randomEffects["gcpStroke"] = self._test_case_one._randomEffects["gcpStroke"] + 5
         self.assertAlmostEqual(
            GCPStrokeModel().get_risk_for_person(self._test_case_one, rng=self._test_case_one._rng, years=1, vectorized=False, test=True),
            self._test_case_one._expectedLinearPredictor+5,
            places=2)

    #def test_randomness_vectorized_independent_per_draw(self):

    #     seedSequence = np.random.SeedSequence()
    #     rngStream = np.random.default_rng(seed=seedSequence)  

    #     #should include both residual and random effect
    #     draw1 = GCPStrokeModel().get_risk_for_person(self._test_case_one_df.iloc[0], rng=rngStream, years=1, vectorized=True, test=False)
    #     draw2 = GCPStrokeModel().get_risk_for_person(self._test_case_one_df.iloc[0], rng=rngStream, years=1, vectorized=True, test=False)
    #     self.assertNotEqual(draw1, draw2)   

if __name__ == "__main__":
    unittest.main()         
