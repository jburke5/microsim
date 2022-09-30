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
        # add a setp to clon people here
        return PersonListPopulation(filteredPeople)

    def randomize(self, randomizationSchema):
        treatedList = []
        untreatedList = []
        for i, person in self.trialPopulation._people.iteritems():
            if randomizationSchema(person):
                treatedList.append(copy.deepcopy(person))
            else:
                untreatedList.append(copy.deepcopy(person))
        return PersonListPopulation(treatedList), PersonListPopulation(untreatedList)
    
    def run(self):
        self.treatedPop._treatments = self.trialDescription._treatments
        self.treatedPop.advance_vectorized(self.trialDescription.duration)
        self.untreatedPop.advance_vectorized(self.trialDescription.duration)
        
    