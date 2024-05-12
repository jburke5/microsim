import numpy as np

from microsim.population_factory import PopulationType
from microsim.trials.trial_type import TrialType
from microsim.treatment_strategy_repository import TreatmentStrategyRepository
from microsim.treatment import TreatmentStrategiesType
from microsim.bp_treatment_strategies import AddNBPMedsTreatmentStrategy

class TrialDescription:
    def __init__(self, 
                 trialType=TrialType.COMPLETELY_RANDOMIZED, 
                 blockFactors=list(),
                 sampleSize=100, 
                 duration=5, 
                 treatmentStrategies=None, 
                 nWorkers=1, 
                 inclusionFilters=None):
        self.trialType = trialType
        self.blockFactors = blockFactors           
        self.sampleSize = sampleSize
        self.duration = duration
        self.treatmentStrategies = self.get_treatment_strategy(treatmentStrategies)
        self.nWorkers = nWorkers
        self.inclusionFilters = inclusionFilters
        self._rng = np.random.default_rng() 
        self.popType = None
        self.is_valid_trial()
        
    def get_treatment_strategy(self, treatmentStrategies):
        if treatmentStrategies is None: 
            return TreatmentStrategyRepository()
        elif type(treatmentStrategies)==str:
            if treatmentStrategies=="1bpMedsAdded":
                ts = TreatmentStrategyRepository()
                ts._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(1)
                return ts
            elif treatmentStrategies=="2bpMedsAdded":
                ts = TreatmentStrategyRepository()
                ts._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(2)
                return ts
            elif treatmentStrategies=="3bpMedsAdded":
                ts = TreatmentStrategyRepository()
                ts._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(3)
                return ts
            elif treatmentStrategies=="4bpMedsAdded":
                ts = TreatmentStrategyRepository()
                ts._repository[TreatmentStrategiesType.BP.value] = AddNBPMedsTreatmentStrategy(4)
                return ts
            else:
                raise RuntimeError("Unrecognized treatmentStrategies argument in TrialDescription initialization.")
        elif type(treatmentStrategies)==TreatmentStrategyRepository:
            return treatmentStrategies
        else:
            raise RuntimeError("Unrecognized treatmentStrategies argument in TrialDescription initialization.")

    def is_valid_trial(self):
        self.assess_trial_type_and_block_factors()
        self.assess_sample_size()
        self.assess_duration()
        self.assess_number_of_workers()   
 
    def is_not_randomized(self):
        return self.trialType==TrialType.NON_RANDOMIZED
       
    def is_block_randomized(self):
        return ((self.trialType==TrialType.COMPLETELY_RANDOMIZED_IN_BLOCKS)|
                (self.trialType==TrialType.BERNOULLI_RANDOMIZED_IN_BLOCKS))
 
    def is_not_block_randomized(self):
        return ((self.trialType==TrialType.COMPLETELY_RANDOMIZED)|
                (self.trialType==TrialType.BERNOULLI_RANDOMIZED))
   
    def is_completely_randomized(self):
         return ((self.trialType==TrialType.COMPLETELY_RANDOMIZED_IN_BLOCKS)|
                (self.trialType==TrialType.COMPLETELY_RANDOMIZED))
        
    def is_bernoulli_randomized(self):
         return ((self.trialType==TrialType.BERNOULLI_RANDOMIZED_IN_BLOCKS)|
                (self.trialType==TrialType.BERNOULLI_RANDOMIZED))

    def assess_trial_type_and_block_factors(self):
        if (self.is_block_randomized()) & (len(self.blockFactors)==0):
            raise RuntimeError("Trial is setup to use blocks but no block factors were provided.")
        elif (not self.is_block_randomized()) & (len(self.blockFactors)>0):
            raise RuntimeError("Trial is not setup to use blocks but block factors were provided.")

    def assess_sample_size(self):
        if (self.sampleSize<=0):
            raise RuntimeError("Sample size cannot be less than or equal to 0.")
        elif  (self.sampleSize>10000000):
            raise RuntimeError("Sample size exceeds the maximum bound.")

    def assess_duration(self):
        if (self.duration<=0):
            raise RuntimeError("Duration cannot be less than or equal to 0.")
        elif  (self.duration>200):
            raise RuntimeError("Duration exceeds the maximum bound.")
            
    def assess_number_of_workers(self):
        if (self.nWorkers<=0):
            raise RuntimeError("Number of workers cannot be less than or equal to 0.")
        elif  (self.nWorkers>100):
            raise RuntimeError("Number of workers exceeds the maximum bound.")
            
    def __str__(self):
        rep = f"Trial Description\n"
        rep += f"\tTrial type: {self.trialType}\n"
        rep += f"\tBlock factors: {self.blockFactors}\n"
        rep += f"\tSample size: {self.sampleSize}\n"
        rep += f"\tDuration: {self.duration}\n"
        rep += f"\tTreatment strategies: {self.treatmentStrategies}\n"
        rep += f"\tNumber of workers: {self.nWorkers}\n"
        rep += f"\tInclusion filters: \n\t {self.inclusionFilters}"
        return rep

    def __repr__(self):
        return self.__str__()

class NhanesTrialDescription(TrialDescription):
    def __init__(self, 
                 trialType=TrialType.COMPLETELY_RANDOMIZED, 
                 blockFactors=list(),
                 sampleSize=100, 
                 duration=5, 
                 treatmentStrategies=TreatmentStrategyRepository(), 
                 nWorkers=1, 
                 inclusionFilters=None,
                 year=1999, 
                 nhanesWeights=False, 
                 distributions=False):
        super().__init__(trialType, blockFactors, sampleSize, duration, treatmentStrategies, nWorkers=nWorkers, inclusionFilters=inclusionFilters)
        self.year = year
        self.nhanesWeights=nhanesWeights
        self.distributions=distributions
        self.popArgs = {"n":self.sampleSize,
                        "year":self.year,
                        "dfFilter":self.inclusionFilters,
                        "nhanesWeights":self.nhanesWeights,
                        "distributions":self.distributions}
        self.popType = PopulationType.NHANES

    def __str__(self):
        rep = super().__str__()
        rep += f"\n\tYear: {self.year}\n"
        rep += f"\tNHANES weights: {self.nhanesWeights}\n"
        rep += f"\tDistributions: {self.distributions}\n"
        rep += f"\tPopulation type: {self.popType}"
        return rep
