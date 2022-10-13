from microsim.population import PersonListPopulation
import copy

class Trial:
    def __init__(self, trialDescription, targetPopulation):
        self.trialDescription = trialDescription
        self.trialPopulation = self.select_trial_population(targetPopulation, 
            trialDescription.inclusionFilter, trialDescription.exclusionFilter)
        # select our patients from the population
        self.treatedPop, self.untreatedPop = self.randomize(trialDescription.randomizationSchema)

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
        #for i in range(0, self.trialDescription.duration):
        #    print(f"************* {i}")
        #    self.treatedDF, self.treatedAlive = self.treatedPop.advance_vectorized(1)
        self.treatedDF, self.treatedAlive = self.treatedPop.advance_vectorized(self.trialDescription.duration)
        self.untreatedDF, self.untreatedAlive = self.untreatedPop.advance_vectorized(self.trialDescription.duration)
        
    