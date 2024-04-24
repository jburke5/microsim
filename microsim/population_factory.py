import pandas as pd

from microsim.person_factory import PersonFactory
from microsim.population import Population
from microsim.afib_model import AFibPrevalenceModel
from microsim.pvd_model import PVDPrevalenceModel
from microsim.risk_factor import DynamicRiskFactorsType
from microsim.population_model_repository import PopulationModelRepository
from microsim.outcome_model_repository import OutcomeModelRepository
from microsim.cohort_risk_model_repository import (CohortDynamicRiskFactorModelRepository, 
                                                   CohortStaticRiskFactorModelRepository,
                                                   CohortDefaultTreatmentModelRepository)

class PopulationFactory:

    @staticmethod
    def get_nhanes_person_initialization_model_repo():
        """Returns the repository needed in order to initialize a Person object using NHANES data.
           This is due to the fact that some risk factors that are needed in Microsim simulations
           are not included in the NHANES data but we have models for these risk factors. """
        return {DynamicRiskFactorsType.AFIB: AFibPrevalenceModel(),
                DynamicRiskFactorsType.PVD: PVDPrevalenceModel()}

    @staticmethod
    def set_index_in_people(people):
        """Once people are created, its Person-objects do not have a unique index.
           This function assigns a unique index to every Person-object in people."""
        list(map(lambda person, i: setattr(person, "_index", i), people, range(people.shape[0])))

    @staticmethod
    def get_nhanes_people(n=None, year=None, dfFilter=None, nhanesWeights=False):
        '''Returns a Pandas Series object with Person-Objects of all persons included in NHANES for year 
           with or without sampling. Filters are applied prior to sampling in order to maximize efficiency and minimize
           memory utilization. This does not affect the distribution of the relative percentages of groups 
           represented in people.'''
        nhanesDf = pd.read_stata("microsim/data/fullyImputedDataset.dta")
        
        if year is not None:
            nhanesDf = nhanesDf.loc[nhanesDf.year == year]

        if dfFilter is not None:
            nhanesDf = nhanesDf.loc[nhanesDf.apply(dfFilter, axis=1)] 

        initializationModelRepository = PopulationFactory.get_nhanes_person_initialization_model_repo()

        #convert the integers to booleans because in the simulation we always use bool for this rf
        nhanesDf["anyPhysicalActivity"] = nhanesDf["anyPhysicalActivity"].astype(bool)

        if nhanesWeights:
            if (year is None) | (n is None):
                raise RuntimeError("""Cannot set nhanesWeights True without specifying a year and n.
                                    NHANES weights are defined for each year independently and for sampling 
                                    to occur the sampling size is needed.""")
            else:
                weights = nhanesDf.WTINT2YR
                nhanesDf = nhanesDf.sample(n, weights=weights, replace=True)
           
        people = pd.DataFrame.apply(nhanesDf,
                                    PersonFactory.get_nhanes_person, initializationModelRepository=initializationModelRepository, axis="columns")

        PopulationFactory.set_index_in_people(people)

        return people

    @staticmethod
    def get_nhanes_population_model_repo():
        """Return the default, self-consistent set of models for advancing an NHANES Population."""
        return PopulationModelRepository(CohortDynamicRiskFactorModelRepository(),
                                         CohortDefaultTreatmentModelRepository(),
                                         OutcomeModelRepository(),
                                         CohortStaticRiskFactorModelRepository())

    @staticmethod    
    def get_nhanes_population(n=None, year=None, dfFilter=None, nhanesWeights=False):
        '''Returns a Population-object with Person-objects being all NHANES persons with or without sampling.'''
        people = PopulationFactory.get_nhanes_people(n=n, year=year, dfFilter=dfFilter, nhanesWeights=nhanesWeights)
        popModelRepository = PopulationFactory.get_nhanes_population_model_repo()
        return Population(people, popModelRepository)


