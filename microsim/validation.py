import numpy as np

from microsim.population_factory import PopulationFactory
from microsim.person_filter_factory import PersonFilterFactory
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.treatment import DefaultTreatmentsType
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial import Trial
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
from microsim.trials.trial_outcome_assessor import AnalysisType
from microsim.trials.trial_type import TrialType
from microsim.outcome import OutcomeType

class Validation:

    @staticmethod
    def nhanes_baseline_pop():
        '''This function performs the simulation for the validation of the creation of the population (baseline models only).'''
        print(f"\nVALIDATION OF BASELINE SIMULATED POPULATION")
        print("2007 Nhanes")
        popSize=100000
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=2007, personFilters=None, nhanesWeights=True, distributions=False)
        pop.print_baseline_summary()
        print("2013 Hypertension")
        pf = PersonFilterFactory.get_person_filter(addCommonFilters=False)
        pf.add_filter(filterType="df",
                      filterName="lowAntiHypertensiveLimit",
                      filterFunction = lambda x: x[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]>0)
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=2013, personFilters=pf, nhanesWeights=True, distributions=False)
        pop.print_baseline_summary()

    @staticmethod
    def nhanes_over_time(nWorkers=5, path=None):
        '''Performs the over time validation of a population against the NHANES sample.
           The filters are used only for the NHANES comparison population from 2017.
           People that died prior to 2017 are not removed from the simulation population, if the simulation population is large enough
           and the death models work well, the resulting simulated population from an advancement of 18 years should be close to the
           NHANES comparison population.
           nWorkers determines the number of cores used
           path=None will result in displaying the figures whereas an actual path will export them to that path'''
        nYears = 18
        popSize = 100000
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=1999, personFilters=None, nhanesWeights=True, distributions=False)
        pop.advance_parallel(nYears, None, nWorkers)
        pf = PersonFilterFactory.get_person_filter(addCommonFilters=False)
        pf.add_filter(filterType="df",
                      filterName="lowAge",
                      filterFunction = lambda x: x[DynamicRiskFactorsType.AGE.value]>=36)
        pf.add_filter(filterType="df",
                      filterName="noImmigration",
                      filterFunction = lambda x: x["timeInUS"]>=4)
        nhanesPop = PopulationFactory.get_nhanes_population(n=popSize, year=2017, personFilters=pf, nhanesWeights=True, distributions=False)

        print("\nVALIDATION OF VASCULAR RISK FACTORS OVER TIME")
        pop.print_vascular_rfs_over_time(nhanesPop, path=path)
        print("\nVALIDATION OF CV EVENT INCIDENCE AND MORTALITY")
        pop.print_cv_standardized_rates()
        print("\nVALIDATION OF DEMENTIA INCIDENCE")
        pop.print_outcome_incidence(path=path, outcomeType=OutcomeType.DEMENTIA)

    @staticmethod
    def nhanes_treatment_effects(sampleSize=2000000, nWorkers=1):
        '''This function creates and advances a control and a treated population in order to estimate the 
           BP medication treatment effect on the MI relative risk and the stroke relative risk.'''
        print("\nVALIDATION OF TREATMENT EFFECTS")
        nYears=5
        nSimulations = 4
        pf = PersonFilterFactory.get_person_filter(addCommonFilters=False)
        for bpMedsAdded in [1,2,3,4]:
            miRRList = list()
            strokeRRList = list()
            print(f"\nbpMedsAdded={bpMedsAdded}")
            for i in range(nSimulations):
                td = NhanesTrialDescription(
                            trialType = TrialType.COMPLETELY_RANDOMIZED,
                            blockFactors=list(),
                            sampleSize = sampleSize,
                            duration = nYears,
                            treatmentStrategies = f"{bpMedsAdded}bpMedsAdded",
                            nWorkers = nWorkers,
                            personFilters=pf,
                            year=1999, nhanesWeights=True, distributions=False)
                toa = TrialOutcomeAssessorFactory.get_trial_outcome_assessor()
                tr = Trial(td)
                tr.run()
                tr.analyze(toa )
                strokeRR = tr.results[AnalysisType.RELATIVE_RISK.value]["strokeRR"][0]
                miRR = tr.results[AnalysisType.RELATIVE_RISK.value]["miRR"][0]
                miRRList += [miRR]
                strokeRRList += [strokeRR]
                print(f"\t\tsimulation={i}, strokeRR= {strokeRR:<8.2f}, miRR= {miRR:<8.2f}")
            print(f"    average of {nSimulations} simulations: strokeRR= {np.mean(strokeRRList):<8.2f}, miRR= {np.mean(miRRList):<8.2f}")
            print(f"         sd of {nSimulations} simulations: strokeRR= {np.std(strokeRRList):<8.2f}, miRR= {np.std(miRRList):<8.2f}")    

    @staticmethod
    def nhanes(path=None):
        Validation.nhanes_baseline_pop()       
        Validation.nhanes_over_time(path=path)
        Validation.nhanes_treatment_effects()



