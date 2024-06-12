from microsim.population_factory import PopulationFactory
from microsim.person_filter_factory import PersonFilterFactory
from microsim.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.treatment import DefaultTreatmentsType
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial import Trial
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
from microsim.trials.trial_type import TrialType

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
    def nhanes_over_time(nWorkers=1, path=None):
        '''Performs the over time validation of a population against the NHANES sample.'''
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
        pop.print_dementia_incidence()

    @staticmethod
    def nhanes_treatment_effects(sampleSize=2000000, nWorkers=5):
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
                strokeRR = tr.results["strokeRR"][0]
                miRR = tr.results["miRR"][0]
                miRRList += [miRR]
                strokeRRList += [strokeRR]
                print(f"\t\tsimulation={i}, strokeRR= {strokeRR:<8.2f}, miRR= {miRR:<8.2f}")
            print(f"average of {nSimulations} simulations: strokeRR= {np.mean(strokeRRList):<12.2f}, miRR= {np.mean(miRRList):8.2f}")
            print(f"sd of {nSimulations} simulations: strokeRR= {np.std(strokeRRList):<21.2f}, miRR= {np.std(miRRList):<8.2f}")    

    @staticmethod
    def nhanes(sampleSize=2000000, nWorkers=5, path=None):
        Validation.nhanes_baseline_pop()       
        Validation.nhanes_over_time(nWorkers=nWorkers, path=path)
        Validation.nhanes_treatment_effects(sampleSize=sampleSize, nWorkers=nWorkers)



