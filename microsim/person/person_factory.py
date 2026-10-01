from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_factors.risk_factor_bounds import RiskFactorBounds
from microsim.outcomes.outcome import OutcomeType
from microsim.person.person import Person
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.common.population_type import PopulationType
from microsim.outcomes.wmh_model_repository import WMHModelRepository
from microsim.outcomes.epilepsy_model import EpilepsyPrevalenceModel
from microsim.outcomes.cognition_model import CognitionPrevalenceModel
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository

#shared by get_kaiser_person so the models are not rebuilt for every person
_kaiserWMHModelRepository = WMHModelRepository()
_kaiserEpilepsyPrevalenceModel = EpilepsyPrevalenceModel()
_kaiserCognitionPrevalenceModel = CognitionPrevalenceModel()

class PersonFactory:
    """A class used to obtain Person-objects using data from a variety of sources."""

    #a dictionary with microsim attributes as keys and dataframe column names as values.
    #This maps microsim standardized person attributes to the non-standardized NHANES dataframe.
    #Useful to convert column names from the NHANES data to the names Microsim uses.
    #We could avoid the use of this dictionary if we could standardize all inputs coming from 
    #dataframes but this could be impossible since some dataframes are created by others, with their own codebooks, column names etc.
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
    def get_person(x, popType=PopulationType.NHANES.value, initializationModelRepository=None, outcomePrevalenceModelRepository=None):
        if popType==PopulationType.NHANES.value:
            return PersonFactory.get_nhanes_person(x, initializationModelRepository, outcomePrevalenceModelRepository=outcomePrevalenceModelRepository)
        elif popType==PopulationType.KAISER.value:
            return PersonFactory.get_kaiser_person(x, initializationModelRepository)
        else:
            raise RuntimeError("Unrecognized population type in PersonFactory.get_person.")

    @staticmethod
    def get_nhanes_person_init_information(x):
        """Takes all Person-instance-related data via x and and organizes it."""

        #x["name"], not x.name: x is a row of a dataframe, ie a Series, and a Series carries a name
        #attribute of its own -- its index label -- which shadows the column of that name. Read as an
        #attribute this silently returned the row's position in the frame rather than its NHANES identifier
        name = x["name"]

        adult = x.age>=18. #need to know for applying the right RiskFactorBounds below
  
        personStaticRiskFactors = {
                            StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity(x.raceEthnicity),
                            StaticRiskFactorsType.EDUCATION.value: Education(x.education),
                            StaticRiskFactorsType.GENDER.value: NHANESGender(x.gender),
                            StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus(x.smokingStatus),
                            StaticRiskFactorsType.MODALITY.value: None}
   
        #TO DO: find a way to include everything here, including the rfs that need initialization
        #the PVD model would be easy to implement, eg with an estimate_next_risk_for_patient_characteristics function
        #but the AFIB model would be more difficult because it relies on the logistic_risk_factor_model file
        #for now include None, in order to create the risk factor lists correctly at the Person instance
        personDynamicRiskFactors = dict()
        for rfd in DynamicRiskFactorsType:
            if rfd==DynamicRiskFactorsType.ALCOHOL_PER_WEEK:
                personDynamicRiskFactors[rfd.value] = float(x[rfd.value]) #drinks/week, no bounds defined
            else:
                if (rfd!=DynamicRiskFactorsType.PVD) & (rfd!=DynamicRiskFactorsType.AFIB):
                    personDynamicRiskFactors[rfd.value] = RiskFactorBounds.apply(rfd.value, x[rfd.value], adult=adult)
        personDynamicRiskFactors[DynamicRiskFactorsType.AFIB.value] = None
        personDynamicRiskFactors[DynamicRiskFactorsType.PVD.value] = None

        #Q: do we need otherLipid treatment? I am not bringing it to the Person objects for now.
        #A: it is ok to leave it out as we do not have a model to update this. It is also very rarely taking place in the population anyway.
        #bool: NHANES includes statin=2 but models expect a 0/1 indicator; matches the Kaiser path
        personDefaultTreatments = {
                            DefaultTreatmentsType.STATIN.value: bool(x.statin),
                            DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: x.antiHypertensiveCount}

        personTreatmentStrategies = dict(zip([strategy.value for strategy in TreatmentStrategiesType],
                                              #[None for strategy in range(len(TreatmentStrategiesType))]))
                                              [{"status": None} for strategy in range(len(TreatmentStrategiesType))]))

        personOutcomes = dict(zip([outcome for outcome in OutcomeType],
                                  [list() for outcome in range(len(OutcomeType))]))

        #priorToSim outcome seeding via selfReport columns is retired in favor of logistic
        #prevalence models registered in OutcomePrevalenceModelRepository, applied in
        #get_nhanes_person via Person.seed_prevalent_outcomes after construction.
        ##If df originates from the NHANES df these columns will exist, but if drawing from the NHANES distributions, these will not be in the df
        #if "selfReportStrokeAge" in x.index:
        #    #add pre-simulation stroke outcomes
        #    selfReportStrokeAge=x.selfReportStrokeAge
        #    #Q: we should not add the stroke outcome in case of "else"? A: No, this is the way it should be
        #    if selfReportStrokeAge is not None and selfReportStrokeAge > 1:
        #        personOutcomes[OutcomeType.STROKE].append((None, StrokeOutcome(False, None, None, None, priorToSim=True)))
        #if "selfReportMIAge" in x.index:
        #    #add pre-simulation mi outcomes
        #    selfReportMIAge=rng.integers(18, x.age) if x.selfReportMIAge == 99999 else x.selfReportMIAge
        #    if selfReportMIAge is not None and selfReportMIAge > 1:
        #        personOutcomes[OutcomeType.MI].append((None, Outcome(OutcomeType.MI, False, priorToSim=True)))

        #if personDynamicRiskFactors[DynamicRiskFactorsType.A1C.value] >= 6.5:
        #    personOutcomes[OutcomeType.DIABETES].append((None, Outcome(OutcomeType.DIABETES, False, priorToSim=True)))

        return (name, personStaticRiskFactors, personDynamicRiskFactors, personDefaultTreatments, personTreatmentStrategies, personOutcomes)

    @staticmethod
    def get_nhanes_person(x, initializationModelRepository, outcomePrevalenceModelRepository=None):
        """Takes all Person-instance-related data via x and initializationModelRepository and organizes it,
           passes the organized data to the Person class and returns a Person instance.
           initializationModelRepository: required. Pass a shared instance when constructing many
           persons (see PopulationFactory.get_nhanes_people); build it via
           InitializationModelRepository() from microsim.risk_factors.initialization_model_repository.
           outcomePrevalenceModelRepository: pass a shared instance when constructing many persons.
           When omitted, priorToSim outcome seeding is skipped."""

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
        person._pvd = [initializationModelRepository[DynamicRiskFactorsType.PVD.value].estimate_next_risk(person)]
        person._afib = [initializationModelRepository[DynamicRiskFactorsType.AFIB.value].estimate_next_risk(person)]
        person._modality = initializationModelRepository[StaticRiskFactorsType.MODALITY.value].estimate_next_risk(person)

        if outcomePrevalenceModelRepository is not None:
            person.seed_prevalent_outcomes(outcomePrevalenceModelRepository)

        return person

    @staticmethod
    def get_kaiser_person_init_information(x):
        name = x["name"]
        personStaticRiskFactors = {
                            StaticRiskFactorsType.MODALITY.value: x.modality,
                            StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity(int(x.raceEthnicity)),
                            StaticRiskFactorsType.EDUCATION.value: None,
                            StaticRiskFactorsType.GENDER.value: NHANESGender(int(x.gender)),
                            StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus(int(x.smokingStatus))}
    
        personDynamicRiskFactors = dict()
        for rfd in DynamicRiskFactorsType:
            if rfd==DynamicRiskFactorsType.ALCOHOL_PER_WEEK:
                personDynamicRiskFactors[rfd.value] = None
            else:
                if (rfd!=DynamicRiskFactorsType.WAIST):
                    personDynamicRiskFactors[rfd.value] = RiskFactorBounds.apply(
                        rfd.value, x[rfd.value], adult=x[DynamicRiskFactorsType.AGE.value]>=18)
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
    def get_kaiser_person(x, initializationModelRepository=None):
        """initializationModelRepository: pass a shared instance when constructing many persons;
           when omitted, one is built for this person alone."""
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
    
        imr = initializationModelRepository if initializationModelRepository is not None \
              else InitializationModelRepository()
        person._waist = [imr[DynamicRiskFactorsType.WAIST.value].estimate_next_risk(person)]
        person._alcoholPerWeek = [imr[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value].estimate_next_risk(person)]
        person._education = imr[StaticRiskFactorsType.EDUCATION.value].estimate_next_risk(person)

        #originally this outcome was obtained along with the rest of the outcomes, however treatment strategies need the CV risk, some of them at least,
        #the CV risks requires knowledge of wmh severity and the rest of the wmh parameters, so I am adding this outcome here... 
        outcome = _kaiserWMHModelRepository.select_outcome_model_for_person(person).get_next_outcome(person)
        person.add_outcome(outcome)

        outcome = _kaiserEpilepsyPrevalenceModel.get_prevalent_outcome(person)
        person.add_outcome(outcome)

        cognitionOutcome = _kaiserCognitionPrevalenceModel.get_prevalent_outcome(person)
        person.add_outcome(cognitionOutcome)

        return person




