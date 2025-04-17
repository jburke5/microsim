from microsim.population_factory import PopulationFactory
from microsim.trials.trial_type import TrialType
from microsim.population import Population
from microsim.treatment import TreatmentStrategiesType, TreatmentStrategyStatus
from microsim.trials.trial_outcome_assessor import AnalysisType

import pandas as pd
import random

class Trial:
    '''This class stores the trial setup, through the TrialDescription instance, trial populations, and trial results.
    treatedPop, controlPop: the two trial populations
    completed: a flag that indicates if the trial has been run (by running I mean whether the populations have been advanced or not)
    results: a dictionary that holds all results obtained with the help of TrialOutcomeAssessor instances.
    A Trial class requires a TrialDescription in order to be initialized.
    Subsequently, a Trial instance can be run.
    A Trial instance by itself does not know how to analyze the trial results, a definition for the trial results does not exist yet.
    The TrialOutcomeAssessor class defines the trial outcomes and links the Trial populations with analysis methodologies.
    An instance of the TrialOutcomeAssessor class is therefore required in order to analyze the results of a Trial instance.'''
    def __init__(self, trialDescription): 
        '''During the initialization of the trial, the populations are obtained.'''
        if trialDescription.popType is None:
            raise RuntimeError(f"popType in trialDescription must belong in the set({[pt for pt in PopulationType]})")
        else:
            self.trialDescription = trialDescription
        self.treatedPop, self.controlPop = self.get_trial_populations()
        self.completed = False
        self.analyzed = False
        self.results = dict()
    
    def get_trial_populations(self):
        '''A Population needs two things: People, PopulationModelRepository.
        The People will be obtained according to the TrialType, the PopulationModelRepository is determined 
        based on the PopulationType (eg for NHANES there is only one self-consistent PopulationModelRepository).'''
        treatedPeople, controlPeople = self.get_trial_people()
        return (Population(treatedPeople, PopulationFactory.get_population_model_repo(self.trialDescription.popType)),
                Population(controlPeople, PopulationFactory.get_population_model_repo(self.trialDescription.popType)))
            
    def get_trial_people(self):
        '''Returns treatedPeople and controlPeople based on TrialType.
        The Person index in treatedPeople and controlPeople is unique for the entire set of treated and control.
        In other words, in the treated+control set of Person objects, there should be a single Person object with index 0.'''
        if self.trialDescription.trialType == TrialType.POTENTIAL_OUTCOMES:
            return self.get_trial_people_identical()
        else: 
            treatedPeople, controlPeople = self.get_trial_people_non_randomized()
            if self.trialDescription.is_not_randomized():
                return treatedPeople, controlPeople
            elif self.trialDescription.is_not_block_randomized():
                people = pd.concat([treatedPeople, controlPeople])
                return self.randomize_trial_people(people)
            elif self.trialDescription.is_block_randomized():
                people = pd.concat([treatedPeople, controlPeople])
                return self.randomize_trial_people_in_blocks(people)
            else:
                raise RuntimeError("Unknown TrialType in Trial.get_trial_people function.")
    
    def get_trial_people_non_randomized(self):
        '''Returns People without performing any kind of process on them.
        Sets a unique index on all Person objects of the trial.'''
        treatedPeople = PopulationFactory.get_people(self.trialDescription.popType, **self.trialDescription.popArgs)
        controlPeople = PopulationFactory.get_people(self.trialDescription.popType, **self.trialDescription.popArgs)
        PopulationFactory.set_index_in_people(controlPeople, start=treatedPeople.shape[0])
        return treatedPeople, controlPeople
    
    def get_trial_people_identical(self):
        '''treated and control people are identical.
        Sets a unique index on all Person objects of the trial.'''
        controlPeople = PopulationFactory.get_people(self.trialDescription.popType, **self.trialDescription.popArgs)
        treatedPeople = Population.get_people_copy(controlPeople)
        PopulationFactory.set_index_in_people(controlPeople, start=treatedPeople.shape[0])
        return treatedPeople, controlPeople
            
    def randomize_trial_people(self, people):
        '''Randomizes people in two groups, treated, control.
        Bernoulli randomization draws from a uniform distribution for each person in the trial, thus the 
        number of treated and control people can and will fluctuate.
        Complete randomization has a fixed number in treated and control people and then randomly assigns 
        groups to the people of the trial.
        Complete randomization has less fluctuations than Bernoulli randomization.'''
        nDraws = people.shape[0]
        if self.trialDescription.is_bernoulli_randomized():
            draws = self.trialDescription._rng.uniform(size=nDraws) 
        elif self.trialDescription.is_completely_randomized():
            draws = [0]*(nDraws//2) + [1]*(nDraws//2) if nDraws%2==0 else [0]*(nDraws//2) + [1]*((nDraws//2)+1)
            draws = random.sample(draws, len(draws))
        else:
            raise RuntimeError("Unknown TrialType in Trial randomize_people function.")
        controlPeople = pd.Series([p for i,p in enumerate(people) if draws[i]<0.5])
        treatedPeople = pd.Series([p for i,p in enumerate(people) if draws[i]>=0.5])
        return treatedPeople, controlPeople
    
    def randomize_trial_people_in_blocks(self, people):
        '''Creates blocks and randomizes people within blocks.
        Because the number of treated and control people can vary within each block,
        eg when the total number of people in a block is odd,
        there may be larger total deviations in the number of Person objects in treated and control people.'''
        blockFactor = self.trialDescription.blockFactors[0]
        blocks = Population.get_people_blocks(people, blockFactor, nBlocks=10)
        categories = blocks.keys()
        treatedPeople = pd.Series(dtype=object)
        controlPeople = pd.Series(dtype=object)
        for cat in categories:
            treatedPeopleBlock, controlPeopleBlock = self.randomize_trial_people(blocks[cat])
            treatedPeople = pd.concat([treatedPeople, treatedPeopleBlock])
            controlPeople = pd.concat([controlPeople, controlPeopleBlock])
        return treatedPeople, controlPeople
        
    def run(self):
        if self.completed:
            print("Cannot run a trial that has already been completed.")
        else:
            #advance control population
            self.controlPop.advance(self.trialDescription.duration, 
                                    treatmentStrategies=None, 
                                    nWorkers=self.trialDescription.nWorkers)
            #advance treated population
            self.treatedPop.advance(1, 
                                    treatmentStrategies = self.trialDescription.treatmentStrategies,
                                    nWorkers=self.trialDescription.nWorkers)
            for key in TreatmentStrategiesType:
                if self.trialDescription.treatmentStrategies._repository[key.value] is not None:
                    self.trialDescription.treatmentStrategies._repository[key.value].status = TreatmentStrategyStatus.MAINTAIN
            self.treatedPop.advance(self.trialDescription.duration-1, 
                                    treatmentStrategies = self.trialDescription.treatmentStrategies,
                                    nWorkers=self.trialDescription.nWorkers)

            self.completed = True
            print("Trial is completed.")
            
    def analyze(self, trialOutcomeAssessor):
        '''Trial outcomes need to be defined in an instance of the TrialOutcomeAssessor class and provided in this function
        in order for the Trial to be able to analyze its populations.'''
        self.analyzed = True
        for assessmentName in trialOutcomeAssessor._assessments.keys():
            assessmentAnalysis = trialOutcomeAssessor._assessments[assessmentName]["assessmentAnalysis"]
            assessmentAnalysisFunction = trialOutcomeAssessor._analysis[assessmentAnalysis]
            assessmentFunctionDict = trialOutcomeAssessor._assessments[assessmentName]["assessmentFunctionDict"]
            assessmentResults = assessmentAnalysisFunction.analyze(self, assessmentFunctionDict, assessmentAnalysis)
            #self.results[assessmentName] = assessmentResults
            if assessmentAnalysis in self.results.keys():
                self.results[assessmentAnalysis][assessmentName] = assessmentResults
            else:
                self.results[assessmentAnalysis] = {assessmentName: assessmentResults}
    
    def run_analyze(self, trialOutcomeAssessor):
        self.run()
        self.analyze(trialOutcomeAssessor)
           
    def print_covariate_distributions(self):
        '''This function is provided to help examine the balance of the Trial populations.'''
        if not self.trialDescription.is_block_randomized():
            print(" "*25, 
                      "self=treated, unique people count= ",  f"{Population.get_unique_people_count(self.treatedPop._people):<8}", 
                      " "*10,
                      "other=control, unique people count= ",  f"{Population.get_unique_people_count(self.controlPop._people):<8}")
            self.treatedPop.print_lastyear_summary_comparison(self.controlPop)
        else:
            blockFactor = self.trialDescription.blockFactors[0]
            people = pd.concat([self.treatedPop._people, self.controlPop._people])
            peopleBlocks = Population.get_people_blocks(people, blockFactor, nBlocks=10)
            for key in peopleBlocks.keys():
                treatedPeopleBlock = pd.Series(list(filter(lambda x: 
                                                           x._index in list(map(lambda x: x._index, self.treatedPop._people)), 
                                                           peopleBlocks[key])))
                controlPeopleBlock = pd.Series(list(filter(lambda x: 
                                                           x._index in list(map(lambda x: x._index, self.controlPop._people)), 
                                                           peopleBlocks[key])))
                print(" "*25, "-"*109)
                print(" "*25, f"block:{blockFactor}={key}")
                print(" "*25, 
                      "self=treated, unique people count=",  Population.get_unique_people_count(treatedPeopleBlock), 
                      " "*15,
                      "other=control, unique people count=",  Population.get_unique_people_count(controlPeopleBlock))
                Population.print_people_summary_at_index_comparison(treatedPeopleBlock, controlPeopleBlock, -1)
             
    def print_treatment_strategy_distributions(self):
        '''Prints distribution information about each treatment strategy variable, eg bpMedsAdded'''
        print(" "*25, "self=treated, unique people count= ",  f"{Population.get_unique_people_count(self.treatedPop._people):<8}")
        self.treatedPop.print_lastyear_treatment_strategy_distributions()

    def __string__(self):
        rep = self.trialDescription.__str__()
        rep += f"\nTrial\n"
        rep += f"\tTrial completed: {self.completed}\n"
        if self.analyzed:
            rep += f"Trial results:\n"
            for analysisType in AnalysisType:
                rep += "\t" + "Analysis: " + f"{analysisType.value}\n"
                if analysisType == AnalysisType.RELATIVE_RISK:
                    rep += "\t" +" "*25 + " "*10 + "relRisk" + " "*4 + "treatedRisk" + " "*4 + "controlRisk" + " "*9 + "|diff|\n"
                else:
                    rep += "\t" +" "*25 + " "*16 + "Z" + " "*6 + "Intercept" + " "*11 + "Z SE" + " "*9 + "pValue\n"
                for key in self.results[analysisType.value].keys():
                    rep += f"\t{key:>25}: "
                    for result in self.results[analysisType.value][key]:
                        if (result is not None) & (result is not float('inf')):
                            rep += f"{result:>15.2f}"
                        elif result== float('inf'):
                            rep += f"{'inf':>15}"
                        else:
                            rep += " "*15
                    rep += "\n"
        return rep

    def __repr__(self):
        return self.__string__()

