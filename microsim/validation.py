import numpy as np
import pandas as pd

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

    @staticmethod
    def kaiser_baseline_pop(wmhSpecific=True):
        print(f"\nVALIDATION OF BASELINE SIMULATED POPULATION\n")
        popSize = 500000
        pop = PopulationFactory.get_kaiser_population(n=popSize, personFilters=None, wmhSpecific=wmhSpecific)
        pop.print_baseline_summary()
        pop.print_wmh_outcome_summary()
        print("\n")
        print(" "*25, "Reference for Kaiser population...")
        print(" "*16, "severity proportion")
        print(" "*25, "-"*20)
        print(f"{'no  0.707':>31}")
        print(f"{'mild  0.172':>31}")
        print(f"{'moderate  0.038':>31}")
        print(f"{'severe  0.015':>31}")
        print(f"{'unknown  0.069':>31}")
        print("\n")
        print(" "*21, "SBI proportion")
        print(" "*25, "-"*20)
        print(f"{'TRUE  0.044':>31}")

    @staticmethod
    def kaiser_over_time(wmhSpecific=True, nWorkers=1):
        print(f"\nVALIDATION OF SIMULATED POPULATION OVER TIME\n")
        print("Note: this function will return a dictionary of Pandas dataframes with the information needed to do a proportional hazards analysis...")
        print("Note: so ensure you will capture the return variable from this function call...")
        print("Note: because this might take a while...")
        popSize = 500000
        pop = PopulationFactory.get_kaiser_population(n=popSize, personFilters=None, wmhSpecific=wmhSpecific)
        pop.advance(11, nWorkers=nWorkers)
        groupStrings = {1:"CT SBI", 2: "CT WMD", 3: "CT BOTH", 0: "CT NONE", 5:"MRI SBI", 6:"MRI WMD", 7:"MRI BOTH", 4:"MRI NONE"}

        ratesRef = {"stroke": 12, "death": 27, "dementia": 11, "mi": 12}
        strokeRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.STROKE], wave=3)
        dementiaRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.DEMENTIA], wave=3)
        deathRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.DEATH], wave=3)
        miRates = pop.get_outcome_incidence_rates_at_end_of_wave(outcomesTypeList=[OutcomeType.MI], wave=3)
        rates = {"stroke": strokeRates, "dementia": dementiaRates, "death": deathRates, "mi": miRates}
        print(" "*12, "Printing outcome incidence rates at the end of year 4...")
        print(" "*12, "References: a Microsim simulation with all WMH-related models.\n")
        print(" "*12, "Outcome     Reference     Simulation")
        print(" "*12, "-"*40)
        for outcome in rates.keys():
            print(" "*10 + f"{outcome:>10}" + f"{ratesRef[outcome]:>14.1f} " + f"{rates[outcome]:>14.1f}")

        print("\n")
        print(" "*12, "Printing outcome incidence rates by SCD group and modality at the end of year 11...")
        print(" "*12, "References: Stroke-Kent2021, Wang2024, Mortality-Clancy2025, Dementia-Kent2022, MI-no available publication.\n")
        print(" "*12, "Mortality rates")
        print(" "*12, "-"*40)
        deathRatesRef = {1:61.5, 2: 63.8, 3: 84.9, 0:18.2, 5:49.2, 6:28.5, 7:53.7, 4:14.}
        deathMinCiRef = {1:59.1, 2:62.6,  3: 80.9, 0:17.8, 5:45.1, 6:27.6, 7:48.8, 4:13.4}
        deathMaxCiRef = {1:63.9, 2:65.1,  3:89.2,  0:18.5, 5:53.6, 6:29.4, 7:59.0, 4:14.6}
        deathRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.DEATH], wave=3)
        deathRatesList = list()
        print("     Group                  Reference     Simulation")
        for group in deathRatesRef.keys():
            deathRatesList += [ [f"{groupStrings[group]:>10} ", 
                                 f"{deathRatesRef[group]:>10.1f} ({deathMinCiRef[group]:>5.1f} - {deathMaxCiRef[group]:>4.1f} ) ",
                                 f"{deathRates[group]:>14.1f}"] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{deathRatesRef[group]:>10.1f} ({deathMinCiRef[group]:>5.1f} - {deathMaxCiRef[group]:>4.1f} ) " +
                  f"{deathRates[group]:>14.1f}")
        print("\n")
        print(" "*12, "Stroke rates")
        print(" "*12, "-"*40)
        strokeRatesRef = {1: 36.6, 2: 28.5, 3: 47.4, 0: 8.2, 5:31.2, 6: 13.,  7:34.5, 4: 4.8}
        strokeMinCiRef = {1: 34.9, 2: 27.7, 3: 44.5, 0: 8.,  5:28.,  6: 12.4, 7:30.6, 4: 4.5}
        strokeMaxCiRef = {1: 38.4, 2: 29.3, 3: 50.5, 0: 8.4, 5:34.6, 6: 13.6, 7:38.7, 4: 5.2}
        strokeRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.STROKE], wave=3)
        strokeRatesList = list()
        print("     Group                  Reference     Simulation")
        for group in strokeRatesRef.keys():
            strokeRatesList += [ [f"{groupStrings[group]:10} ", 
                                  f"{strokeRatesRef[group]:>4.1f} ({strokeMinCiRef[group]:>5.1f} - {strokeMaxCiRef[group]:>4.1f} ) ",
                                  f"{strokeRates[group]:<4.1f}" ] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{strokeRatesRef[group]:>10.1f} ({strokeMinCiRef[group]:>5.1f} - {strokeMaxCiRef[group]:>4.1f} ) " +
                  f"{strokeRates[group]:>14.1f}")
        print("\n")
        print(" "*12, "MI rates")
        print(" "*12, "-"*40)
        miRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.MI], wave=3)
        print("     Group                                Simulation")
        miRatesList = list()
        for group in groupStrings.keys():
            miRatesList += [ [f"{groupStrings[group]:>10} ",  
                              f"{miRates[group]:>14.1f}"] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{miRates[group]:>41.1f}")
        print("\n")
        print(" "*12, "Dementia rates")
        print(" "*12, "-"*40)
        dementiaRatesRef = {1:32.8, 2:37.7, 3:51.6, 0:6.7, 5:16.6, 6:9.6, 7:19.1, 4:2.9}
        dementiaMinCiRef = {1:31.,  2:36.7, 3:48.3, 0:6.5, 5:14.2, 6:9.1, 7:16.2, 4:2.7}
        dementiaMaxCiRef = {1:34.6, 2:38.7, 3:55.1, 0:6.9, 5:19.3, 6:10.1,7:22.4, 4:3.3}
        dementiaRates = pop.get_outcome_incidence_rates_by_scd_and_modality_at_end_of_wave(outcomesTypeList=[OutcomeType.DEMENTIA], wave=3)
        dementiaRatesList = list()
        print("     Group                  Reference     Simulation")
        for group in dementiaRatesRef.keys():
            dementiaRatesList += [ [f"{groupStrings[group]:>10} ", 
                                    f"{dementiaRatesRef[group]:>10.1f} ({dementiaMinCiRef[group]:>5.1f} - {dementiaMaxCiRef[group]:>4.1f} ) ",
                                    f"{dementiaRates[group]:>14.1f}"] ]
            print(f"{groupStrings[group]:>10} " + 
                  f"{dementiaRatesRef[group]:>10.1f} ({dementiaMinCiRef[group]:>5.1f} - {dementiaMaxCiRef[group]:>4.1f} ) " +
                  f"{dementiaRates[group]:>14.1f}")

        #obtain data for the stroke survival analysis, see figure 1 in Kent2021
        strokeInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.STROKE],
                                                   personFunctionsList = [lambda x: x.get_scd_group(), 
                                                                          lambda x: x.get_wmh_severity_by_modality_group()])
        strokeDf = pd.DataFrame(strokeInfo, columns=["time","event", "sbiwmhGroup", "severityGroup"])
  
        miInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.MI],
                                               personFunctionsList = [lambda x: x.get_scd_group(), 
                                                                      lambda x: x.get_wmh_severity_by_modality_group()])
        miDf = pd.DataFrame(miInfo, columns=["time","event", "sbiwmhGroup", "severityGroup"])

        #obtain data for the dementia survival analysis, see figure 2 in Kent2023
        dementiaInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.DEMENTIA],
                                                     personFunctionsList = [lambda x: x.get_wmh_severity_by_modality_group(),
                                                                            lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "sbi")),
                                                                            lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "wmh"))])
        dementiaDf = pd.DataFrame(dementiaInfo, columns=["time","event", "severityGroup", "sbi", "wmh"])

        deathInfo = pop.get_outcome_survival_info(outcomesTypeList = [OutcomeType.DEATH],
                                                  personFunctionsList = [lambda x: x.get_wmh_severity_by_modality_group(),
                                                                         lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "sbi")),
                                                                         lambda x: int(x.get_outcome_item_first(OutcomeType.WMH, "wmh"))])
        deathDf = pd.DataFrame(deathInfo, columns=["time","event", "severityGroup", "sbi", "wmh"])

        return {"death": deathDf, "mi": miDf, "stroke": strokeDf, "dementia": dementiaDf}

    @staticmethod
    def kaiser(wmhSpecific=True, nWorkers=1):
        Validation.kaiser_baseline_pop(wmhSpecific=wmhSpecific)
        dfs = Validation.kaiser_over_time(wmhSpecific=wmhSpecific, nWorkers=nWorkers)
        return dfs

 

