import numpy as np

from microsim.population_factory import PopulationType
from microsim.trials.trial_type import TrialType
from microsim.treatment_strategy_repository import TreatmentStrategyRepository

class TrialDescription:
    def __init__(self, 
                 trialType=TrialType.COMPLETELY_RANDOMIZED, 
                 blockFactors=list(),
                 sampleSize=100, 
                 duration=5, 
                 treatmentStrategies=TreatmentStrategyRepository(), 
                 nWorkers=1, 
                 inclusionFilters=None):
        self.trialType = trialType
        self.blockFactors = blockFactors           
        self.sampleSize = sampleSize
        self.duration = duration
        self.treatmentStrategies = treatmentStrategies
        self.nWorkers = nWorkers
        self.inclusionFilters = inclusionFilters
        self._rng = np.random.default_rng() 
        self.popType = None
        self.is_valid_trial()
        
    def is_valid_trial(self):
        self.assess_trial_type_and_block_factors()
        self.assess_sample_size()
        self.assess_duration()
        self.assess_number_of_workers()   
        
    def is_block_randomized(self):
        return ((self.trialType==TrialType.COMPLETELY_RANDOMIZED_IN_BLOCKS)|
                (self.trialType==TrialType.BERNOULLI_RANDOMIZED_IN_BLOCKS))
    
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
