from microsim.population import PersonListPopulation
from microsim.trials.trial_utils import get_analysis_name
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import copy
import pandas as pd
import numpy as np

class Trial:
    
    def __init__(self, trialDescription, targetPopulation, rng=None, additionalLabels=None): 
        self.trialDescription = trialDescription
        self.trialPopulation = self.select_trial_population(targetPopulation, trialDescription.inclusionFilters, trialDescription.exclusionFilters)
        # select our patients from the population
        self.maxSampleSize = pd.Series(trialDescription.sampleSizes).max()
        self.treatedPop, self.untreatedPop = self.randomize(trialDescription.randomizationSchema, rng)
        self.analyticResults = {}
        self.additionalLabels = additionalLabels 

    def select_trial_population(self, targetPopulation, inclusionFilters, exclusionFilters): #inclusionFilters: a list of filters
        filteredPeople = list(targetPopulation._people) #initialize
        for inclusionFilter in inclusionFilters:
            filteredPeople = list(filter(inclusionFilter, filteredPeople)) #applies logical AND to all inclusion filters provided
        return PersonListPopulation(filteredPeople)

    def randomize(self, randomizationSchema, rng=None):
        #rng = np.random.default_rng(rng)
        treatedList = []
        untreatedList = []
        # might be able to make this more efficient by sampling from the filtered people...
        for i, person in self.trialPopulation._people.items():
            if not person.is_dead():
                if randomizationSchema(person, rng):
                    treatedList.append(copy.deepcopy(person))
                else:
                    untreatedList.append(copy.deepcopy(person))
            else:
                continue
        return PersonListPopulation(treatedList), PersonListPopulation(untreatedList)

    def run(self, rng=None):
        #rng = np.random.default_rng(rng)
        self.treatedPop._bpTreatmentStrategy = self.trialDescription.treatment
        lastDuration = 0
        for duration in self.trialDescription.durations:
            self.treatedDF, self.treatedAlive = self.treatedPop.advance_vectorized(duration-lastDuration, rng)
            self.untreatedDF, self.untreatedAlive = self.untreatedPop.advance_vectorized(duration-lastDuration, rng)
            self.analyze(duration, 
                         self.maxSampleSize, 
                         self.treatedPop._people.tolist(), 
                         self.untreatedPop._people.tolist(),
                         sampleSizeIndex=0) #there is only one maxSampleSize
            self.analyzeSmallerTrials(duration)
            lastDuration = duration

    def analyzeSmallerTrials(self, duration):
        for sampleSize in self.trialDescription.sampleSizes:
            numTrialsForSample = self.maxSampleSize // sampleSize
            for i in range(0, numTrialsForSample):
                if sampleSize==self.maxSampleSize:
                    continue
                sampleTreated = self.treatedPop._people.sample(int(sampleSize/2))
                sampleUntreated = self.untreatedPop._people.sample(int(sampleSize/2))
                self.analyze(duration, sampleSize, sampleTreated.tolist(), sampleUntreated.tolist(), sampleSizeIndex=i)

    def analyze(self, duration, sampleSize, treatedPopList, untreatedPopList, sampleSizeIndex=0):
        totalBPMedsAddedTreated = sum([sum(x._bpMedsAdded) for x in treatedPopList])
        totalBPMedsAddedUntreated = sum([sum(x._bpMedsAdded) for x in untreatedPopList])
        uniqueIndicesTreated = len(set([x.dfIndex for x in treatedPopList]))
        uniqueIndicesUntreated = len(set([x.dfIndex for x in untreatedPopList]))
        for analysis in self.trialDescription.analyses:
            reg, intercept, se, pvalue, meanUntreated, meanTreated = None, None, None, None, None, None
            try: #get_means returns both meanUntreated and meanTreated, in this order, hence the parenthesis
                reg, intercept, se, pvalue, (meanUntreated, meanTreated) = analysis.analyze(treatedPopList, untreatedPopList)
            except PerfectSeparationError: # how to track these is not obvious, now now we'll enter "Nones"
                pass
            except np.linalg.LinAlgError:
                pass
            self.analyticResults[get_analysis_name(analysis, duration, sampleSize, sampleSizeIndex=sampleSizeIndex)] = {  'reg' : reg,
                                                                                         'se' : se,
                                                                                         'pvalue': pvalue,
                                                                                         'intercept' : intercept,
                                                                                         'meanUntreated' : meanUntreated,
                                                                                         'meanTreated' : meanTreated,
                                                                                         'totalBPMedsAddedUntreated' : totalBPMedsAddedUntreated,
                                                                                         'totalBPMedsAddedTreated' : totalBPMedsAddedTreated,
                                                                                         'uniqueIndicesUntreated' : uniqueIndicesUntreated,
                                                                                         'uniqueIndicesTreated' : uniqueIndicesTreated,
                                                                                         'duration' : duration,
                                                                                         'sampleSize' : sampleSize,
                                                                                         'outcome' :  analysis.outcomeAssessor.get_name(),
                                                                                         'analysis' : analysis.name}
        return self.analyticResults


