import numpy as np
import pandas as pd

from microsim.alcohol_category import AlcoholCategory
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_model_repository import RiskModelRepository
from microsim.outcome import Outcome, OutcomeType
from microsim.person import Person
from microsim.race_ethnicity import RaceEthnicity
from microsim.education import Education
from microsim.gender import NHANESGender
from microsim.smoking_status import SmokingStatus
from microsim.treatment import DefaultTreatmentsType, TreatmentStrategiesType
from microsim.stroke_outcome import StrokeOutcome
from microsim.afib_model import AFibPrevalenceModel
from microsim.pvd_model import PVDPrevalenceModel
from microsim.waist_model import WaistPrevalenceModel
from microsim.education_model import EducationPrevalenceModel
from microsim.alcohol_model import AlcoholPrevalenceModel
from microsim.population_type import PopulationType
from microsim.modality_model import ModalityPrevalenceModel
from microsim.wmh_model_repository import WMHModelRepository

class PersonFactory:
    """A class used to obtain Person-objects using data from a variety of sources."""

    #a dictionary with microsim attributes as keys and dataframe column names as values.
    #Useful to convert column names from the NHANES data to the names Microsim uses.'''
    #Q: this probably belongs somewhere else...but I also need to avoid circular imports...
    microsimToNhanes = {DynamicRiskFactorsType.SBP.value: "meanSBP",
                    DynamicRiskFactorsType.DBP.value: "meanDBP",
                    DynamicRiskFactorsType.A1C.value: "a1c",
                    DynamicRiskFactorsType.HDL.value: "hdl",
                    DynamicRiskFactorsType.LDL.value: "ldl",
                    DynamicRiskFactorsType.TRIG.value: "trig",
                    DynamicRiskFactorsType.TOT_CHOL.value: "tot_chol",
                    DynamicRiskFactorsType.BMI.value: "bmi",
                    DynamicRiskFactorsType.WAIST.value: "waist",
                    DynamicRiskFactorsType.AGE.value: "age",
                    DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: 'anyPhysicalActivity',
                    DynamicRiskFactorsType.CREATININE.value: "serumCreatinine",
                    DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: "alcoholPerWeek",
                    DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: "antiHypertensive"} 

    #same thing for Kaiser data
    microsimToKaiser = {StaticRiskFactorsType.MODALITY.value: "Modality",
                    StaticRiskFactorsType.GENDER.value: "Gender",
                    StaticRiskFactorsType.RACE_ETHNICITY.value: "Race_ETH",
                    StaticRiskFactorsType.SMOKING_STATUS.value: "Tobacco_Ever",
                    DynamicRiskFactorsType.AFIB.value: "Afib",
                    DynamicRiskFactorsType.PVD.value: "PVD",
                    DefaultTreatmentsType.STATIN.value:  "Statins",
                    DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: "anyPhysicalActivity",
                    DynamicRiskFactorsType.AGE.value:  "Age",
                    DynamicRiskFactorsType.HDL.value: "HDL",
                    DynamicRiskFactorsType.A1C.value: "H1A1c",
                    DynamicRiskFactorsType.TOT_CHOL.value: "TotCholesterol",
                    DynamicRiskFactorsType.LDL.value: "LDL",
                    DynamicRiskFactorsType.TRIG.value: "Triglycerides",
                    DynamicRiskFactorsType.CREATININE.value: "Creatinine",
                    DynamicRiskFactorsType.SBP.value:  "SBP",
                    DynamicRiskFactorsType.DBP.value:  "DBP",
                    DynamicRiskFactorsType.BMI.value: "BMI",
                    DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: "N_AntiHTNperYR"}

    @staticmethod
    def get_person(x, popType=PopulationType.NHANES.value):
        if popType==PopulationType.NHANES.value:
            return PersonFactory.get_nhanes_person(x)
        elif popType==PopulationType.KAISER.value:
            return PersonFactory.get_kaiser_person(x)
        else:
            raise RuntimeError("Unrecognized population type in PersonFactory.get_person.")

    @staticmethod
    def get_nhanes_person_init_information(x):
        """Takes all Person-instance-related data via x and and organizes it."""

        rng = np.random.default_rng()

        name = x.name
   
        personStaticRiskFactors = {
                            StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity(int(x.raceEthnicity)),
                            StaticRiskFactorsType.EDUCATION.value: Education(int(x.education)),
                            StaticRiskFactorsType.GENDER.value: NHANESGender(int(x.gender)),
                            StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus(int(x.smokingStatus)),
                            StaticRiskFactorsType.MODALITY.value: None}
   
        #use this to get the bounds imposed on the risk factors in a bit
        rfRepository = RiskModelRepository()

        #TO DO: find a way to include everything here, including the rfs that need initialization
        #the PVD model would be easy to implement, eg with an estimate_next_risk_for_patient_characteristics function
        #but the AFIB model would be more difficult because it relies on statsmodel_logistic_risk file
        #for now include None, in order to create the risk factor lists correctly at the Person instance
        personDynamicRiskFactors = dict()
        for rfd in DynamicRiskFactorsType:
            if rfd==DynamicRiskFactorsType.ALCOHOL_PER_WEEK:
                personDynamicRiskFactors[rfd.value] = AlcoholCategory(x[rfd.value])
            else:
                if (rfd!=DynamicRiskFactorsType.PVD) & (rfd!=DynamicRiskFactorsType.AFIB):
                    personDynamicRiskFactors[rfd.value] = rfRepository.apply_bounds(rfd.value, x[rfd.value])
        personDynamicRiskFactors[DynamicRiskFactorsType.AFIB.value] = None
        personDynamicRiskFactors[DynamicRiskFactorsType.PVD.value] = None

        #Q: do we need otherLipid treatment? I am not bringing it to the Person objects for now.
        #A: it is ok to leave it out as we do not have a model to update this. It is also very rarely taking place in the population anyway.
        #also: used to have round(x.statin) but NHANES includes statin=2...
        personDefaultTreatments = {
                            DefaultTreatmentsType.STATIN.value: bool(x.statin),
                            #DefaultTreatmentsType.OTHER_LIPID_LOWERING_MEDICATION_COUNT.value: x.otherLipidLowering,
                            DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: x.antiHypertensiveCount}

        personTreatmentStrategies = dict(zip([strategy.value for strategy in TreatmentStrategiesType],
                                              #[None for strategy in range(len(TreatmentStrategiesType))]))
                                              [{"status": None} for strategy in range(len(TreatmentStrategiesType))]))

        personOutcomes = dict(zip([outcome for outcome in OutcomeType],
                                  [list() for outcome in range(len(OutcomeType))]))

        #If df originates from the NHANES df these columns will exist, but if drawing from the NHANES distributions, these will not be in the df
        if "selfReportStrokeAge" in x.index:
            #add pre-simulation stroke outcomes
            selfReportStrokeAge=x.selfReportStrokeAge
            #Q: we should not add the stroke outcome in case of "else"? A: No, this is the way it should be
            if selfReportStrokeAge is not None and selfReportStrokeAge > 1:
                selfReportStrokeAge = selfReportStrokeAge if selfReportStrokeAge <= x.age else x.age
                personOutcomes[OutcomeType.STROKE].append((selfReportStrokeAge, StrokeOutcome(False, None, None, None, priorToSim=True)))
        if "selfReportMIAge" in x.index:
            #add pre-simulation mi outcomes
            selfReportMIAge=rng.integers(18, x.age) if x.selfReportMIAge == 99999 else x.selfReportMIAge
            if selfReportMIAge is not None and selfReportMIAge > 1:
                selfReportMIAge = selfReportMIAge if selfReportMIAge <= x.age else x.age
                personOutcomes[OutcomeType.MI].append((selfReportMIAge, Outcome(OutcomeType.MI, False, priorToSim=True)))

        return (name, personStaticRiskFactors, personDynamicRiskFactors, personDefaultTreatments, personTreatmentStrategies, personOutcomes)

    @staticmethod
    def get_nhanes_person(x):
        """Takes all Person-instance-related data via x and initializationModelRepository and organizes it,
           passes the organized data to the Person class and returns a Person instance."""

        (name, 
         personStaticRiskFactors, 
         personDynamicRiskFactors, 
         personDefaultTreatments, 
         personTreatmentStrategies, 
         personOutcomes) = PersonFactory.get_nhanes_person_init_information(x)

        person = Person(name,
                        personStaticRiskFactors,
                        personDynamicRiskFactors,
                        personDefaultTreatments,
                        personTreatmentStrategies,
                        personOutcomes)

        #TO DO: find a way to initialize these rfs above with everything else
        imr = PersonFactory.initialization_model_repository()
        person._pvd = [imr[DynamicRiskFactorsType.PVD.value].estimate_next_risk(person)]
        person._afib = [imr[DynamicRiskFactorsType.AFIB.value].estimate_next_risk(person)]
        person._modality = imr[StaticRiskFactorsType.MODALITY.value].estimate_next_risk(person)
        return person

    @staticmethod
    def initialization_model_repository():
        """Returns the repository needed in order to initialize a Person object.
           This is due to the fact that some risk factors that are needed in Microsim simulations
           are not included in the data we use to construct persons but we have models for these risk factors. """
        return {DynamicRiskFactorsType.AFIB.value: AFibPrevalenceModel(),
                DynamicRiskFactorsType.PVD.value: PVDPrevalenceModel(),
                DynamicRiskFactorsType.WAIST.value: WaistPrevalenceModel(),
                StaticRiskFactorsType.EDUCATION.value: EducationPrevalenceModel(),
                DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholPrevalenceModel(),
                StaticRiskFactorsType.MODALITY.value: ModalityPrevalenceModel()}

    @staticmethod
    def get_kaiser_person_init_information(x):
        name = x["name"]
        personStaticRiskFactors = {
                            StaticRiskFactorsType.MODALITY.value: x.modality,
                            StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity(int(x.raceEthnicity)),
                            StaticRiskFactorsType.EDUCATION.value: None,
                            StaticRiskFactorsType.GENDER.value: NHANESGender(int(x.gender)),
                            StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus(int(x.smokingStatus))}
    
        rfRepository = RiskModelRepository()
    
        personDynamicRiskFactors = dict()
        for rfd in DynamicRiskFactorsType:
            if rfd==DynamicRiskFactorsType.ALCOHOL_PER_WEEK:
                personDynamicRiskFactors[rfd.value] = None
            else:
                if (rfd!=DynamicRiskFactorsType.WAIST):
                    personDynamicRiskFactors[rfd.value] = rfRepository.apply_bounds(rfd.value, x[rfd.value])
        personDynamicRiskFactors[DynamicRiskFactorsType.WAIST.value] = None
    
        personDefaultTreatments = {
                            DefaultTreatmentsType.STATIN.value: bool(x.statin),
                            DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: x.antiHypertensiveCount}
    
        personTreatmentStrategies = dict(zip([strategy.value for strategy in TreatmentStrategiesType],
                                              #[None for strategy in range(len(TreatmentStrategiesType))]))
                                              [{"status": None} for strategy in range(len(TreatmentStrategiesType))]))
    
        personOutcomes = dict(zip([outcome for outcome in OutcomeType],
                                  [list() for outcome in range(len(OutcomeType))]))
    
        return (name, personStaticRiskFactors, personDynamicRiskFactors, personDefaultTreatments, personTreatmentStrategies, personOutcomes)

    @staticmethod
    def get_kaiser_person(x):
        (name, 
         personStaticRiskFactors, 
         personDynamicRiskFactors, 
         personDefaultTreatments, 
         personTreatmentStrategies, 
         personOutcomes) = PersonFactory.get_kaiser_person_init_information(x)

        person = Person(name,
                        personStaticRiskFactors,
                        personDynamicRiskFactors,
                        personDefaultTreatments,
                        personTreatmentStrategies,
                        personOutcomes)
    
        imr = PersonFactory.initialization_model_repository()
        person._waist = [imr[DynamicRiskFactorsType.WAIST.value].estimate_next_risk(person)]
        person._alcoholPerWeek = [imr[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value].estimate_next_risk(person)]
        person._education = imr[StaticRiskFactorsType.EDUCATION.value].estimate_next_risk(person)

        #originally this outcome was obtained along with the rest of the outcomes, however treatment strategies need the CV risk, some of them at least,
        #the CV risks requires knowledge of wmh severity and the rest of the wmh parameters, so I am adding this outcome here... 
        outcome = WMHModelRepository().select_outcome_model_for_person(person).get_next_outcome(person)
        person.add_outcome(outcome)
        
        return person




