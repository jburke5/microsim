from microsim.population import PersonListPopulation
import copy
import pandas as pd

class Trial:
    def __init__(self, trialDescription, targetPopulation):
        self.trialDescription = trialDescription
        self.trialPopulation = self.select_trial_population(targetPopulation, 
            trialDescription.inclusionFilter, trialDescription.exclusionFilter)
        # select our patients from the population
        self.treatedPop, self.untreatedPop = self.randomize(trialDescription.randomizationSchema)
        self.analyticResults = {}

    def select_trial_population(self, targetPopulation, inclusionFilter, exclusionFilter):
        filteredPeople = list(filter(inclusionFilter, list(targetPopulation._people)))
        # add a setp to clone people here
        return PersonListPopulation(filteredPeople)

    def randomize(self, randomizationSchema):
        treatedList = []
        untreatedList = []
        randomizedCount = 0
        # might be able to make this more efficient by sampling from the filtered people...
        for i, person in self.trialPopulation._people.iteritems():
            while randomizedCount < self.trialDescription.sampleSize: 
                if not person.is_dead():
                    if randomizationSchema(person):
                        treatedList.append(copy.deepcopy(person))
                    else:
                        untreatedList.append(copy.deepcopy(person))
                    randomizedCount+=1
                else:
                    continue 
        return PersonListPopulation(treatedList), PersonListPopulation(untreatedList)
    
    def run(self):
        self.treatedPop._bpTreatmentStrategy = self.trialDescription.treatment
        lastDuration = 0
        for duration in self.trialDescription.durations:
            self.treatedDF, self.treatedAlive = self.treatedPop.advance_vectorized(duration-lastDuration)
            self.untreatedDF, self.untreatedAlive = self.untreatedPop.advance_vectorized(duration-lastDuration)
            self.analyze(duration)
            lastDuration = duration


    def analyze(self, duration):
        for analysis in self.trialDescription.analyses:
            reg, se, pvalue = analysis.analyze(self.treatedPop, self.untreatedPop)
            self.analyticResults[get_analysis_name(analysis, duration)] = {'reg' : reg, 'se' : se, 'pvalue': pvalue}
        return self.analyticResults
        
        
def get_analysis_name(analysis, duration):
    return f"{analysis.get_name()}-{str(duration)}"
    