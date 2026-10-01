import numpy as np
import pandas as pd

from microsim.population.population_factory import PopulationFactory
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.trials.trial_description import NhanesTrialDescription
from microsim.trials.trial import Trial
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
from microsim.trials.trial_outcome_assessor import AnalysisType
from microsim.trials.trial_type import TrialType
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.reference import Reference

class Burke2024:
    '''The published values that Burke et al., "Development and validation of the Michigan Chronic
       Disease Simulation Model (MICROSIM)", PLOS ONE 19(5):e0300005 (2024), validated MICROSIM
       against, so that a run of the NHANES validation can be printed next to them.
       Every value here comes from outside MICROSIM: survey-weighted NHANES estimates (Table 2 and
       the text describing Fig 2), population-based incidence studies and US population mortality
       (Table 3), and a meta-analysis of BP-lowering trials.'''

    #Table 2, the survey-weighted NHANES 2007-2010 cohort.
    baseline2007 = {
        "age, years (mean)": 45.9,
        "female (%)": 51.7,
        "white (%)": 68.4,
        "black (%)": 11.5,
        "hispanic (%)": 13.6,
        "BMI, kg/m2 (mean)": 28.5,
    }

    #Table 2, the survey-weighted NHANES 2013 cohort with hypertension, defined there as
    #SBP > 140/90 mm Hg or taking any anti-hypertensive medication.
    baseline2013Hypertension = {
        "age, years (mean)": 60.0,
        "female (%)": 50.0,
        "white (%)": 71.0,
        "black (%)": 14.0,
        "hispanic (%)": 10.0,
        "BMI, kg/m2 (mean)": 31.0,
        "SBP, mm Hg (mean)": 133.4,
        "DBP, mm Hg (mean)": 71.6,
        "anti-hypertensive use (%)": 41.0,
        "statin use (%)": 41.0,
    }

    #Fig 2 carries no numbers, but the text describing it quotes the DBP and total cholesterol levels of the NHANES 2017 pseudo-cohort 
    #that the 18-year simulation was compared against.
    nhanes2017 = {
        (DynamicRiskFactorsType.DBP.value, "mean"): 71.6,
        (DynamicRiskFactorsType.DBP.value, "sd"): 10.9,
        (DynamicRiskFactorsType.TOT_CHOL.value, "mean"): 196.8,
        (DynamicRiskFactorsType.TOT_CHOL.value, "sd"): 41.3,
    }

    #Table 3 and the mortality sentence that follows it. Events per 100,000 population per year age-sex standardized. The two all-race estimates are published as ranges.
    cvIncidence = {
        "MI incidence, all (per 100,000)": (208., 284.),
        "MI incidence, white (per 100,000)": 199.,
        "MI incidence, black (per 100,000)": 189.,
        "stroke incidence, all (per 100,000)": (130., 400.),
        "stroke incidence, white (per 100,000)": 208.,
        "stroke incidence, black (per 100,000)": 331.,
        "mortality, all (per 100,000)": 729.,
    }

    #Relative risks per added BP medication, 
    treatmentEffects = {
        "stroke RR, 1 BP medication added": 0.79,
        "MI RR, 1 BP medication added": 0.87,
    }

class Validation:

    @staticmethod
    def _percent_of(proportions, categoryValues):
        '''Returns, as a percentage, the share of a proportions dictionary (see
           Population.get_summary_at_index) held by the given category values. Keys are compared
           as ints, so the enum members the proportions are keyed by and plain ints both match.'''
        return 100. * sum([prop for key, prop in proportions.items()
                           if int(key) in categoryValues])

    @staticmethod
    def _baseline_metrics(pop):
        '''Returns the quantities Burke2024 tabulates in Table 2 for a population at baseline, in
           the units the paper uses: means for the continuous variables, percentages for the
           categorical ones. Keyed by the metric names of the Burke2024 reference dictionaries.'''
        summary = pop.get_summary_at_index(0)
        continuous = summary["continuous"]
        proportions = summary["proportions"]
        #anti-hypertensives are a continuous treatment (a count), so the share of people taking any
        #of them is not in the proportions of the summary and is obtained from the counts directly
        antiHypertensiveCounts = pop.get_attr_at_index(DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value, 0)
        return {
            "age, years (mean)": continuous[DynamicRiskFactorsType.AGE.value]["mean"],
            "BMI, kg/m2 (mean)": continuous[DynamicRiskFactorsType.BMI.value]["mean"],
            "SBP, mm Hg (mean)": continuous[DynamicRiskFactorsType.SBP.value]["mean"],
            "DBP, mm Hg (mean)": continuous[DynamicRiskFactorsType.DBP.value]["mean"],
            "female (%)": Validation._percent_of(proportions[StaticRiskFactorsType.GENDER.value],
                                                 [NHANESGender.FEMALE]),
            "white (%)": Validation._percent_of(proportions[StaticRiskFactorsType.RACE_ETHNICITY.value],
                                                [RaceEthnicity.NON_HISPANIC_WHITE]),
            "black (%)": Validation._percent_of(proportions[StaticRiskFactorsType.RACE_ETHNICITY.value],
                                                [RaceEthnicity.NON_HISPANIC_BLACK]),
            "hispanic (%)": Validation._percent_of(proportions[StaticRiskFactorsType.RACE_ETHNICITY.value],
                                                   [RaceEthnicity.MEXICAN_AMERICAN, RaceEthnicity.OTHER_HISPANIC]),
            "anti-hypertensive use (%)": float(100. * np.mean([count>0 for count in antiHypertensiveCounts])),
            "statin use (%)": Validation._percent_of(proportions[DefaultTreatmentsType.STATIN.value], [1]),
        }

    @staticmethod
    def nhanes_baseline_pop():
        '''This function performs the simulation for the validation of the creation of the population (baseline models only).
           Returns {"2007": metrics, "2013Hypertension": metrics}, the Table 2 quantities of each of
           the two populations (see _baseline_metrics), for print_baseline_pop_with_burke2024.'''
        print(f"\nVALIDATION OF BASELINE SIMULATED POPULATION")
        print("2007 Nhanes")
        popSize=100000
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=2007, personFilters=None, nhanesWeights=True, distributions=False)
        pop.print_baseline_summary()
        metrics2007 = Validation._baseline_metrics(pop)
        print("2013 Hypertension")
        #hypertension as Burke2024 defines it in Table 2: SBP > 140/90 mm Hg or on any
        #anti-hypertensive medication. Selecting only those on medication would make a population
        #that is 100% medicated, which is not the population the published estimates describe.
        pf = PersonFilterFactory.get_person_filter(["adult"])
        pf.add_filter(filterType="df",
                      filterName="hypertension",
                      filterFunction = lambda x: ((x[DynamicRiskFactorsType.SBP.value]>140) |
                                                  (x[DynamicRiskFactorsType.DBP.value]>90) |
                                                  (x[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value]>0)))
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=2013, personFilters=pf, nhanesWeights=True, distributions=False)
        pop.print_baseline_summary()
        return {"2007": metrics2007, "2013Hypertension": Validation._baseline_metrics(pop)}

    @staticmethod
    def nhanes_over_time(nWorkers=5, path=None, distributions=False):
        '''Performs the over time validation of a population against the NHANES sample.
           The advanced 1999 population excludes people with a history of stroke or MI, like the
           Burke2024 cohort; the immigration/age filters apply only to the NHANES 2017 comparison
           population.
           People that died prior to 2017 are not removed from the simulation population, if the simulation population is large enough
           and the death models work well, the resulting simulated population from an advancement of 18 years should be close to the
           NHANES comparison population.
           nWorkers determines the number of cores used
           path=None will result in displaying the figures whereas an actual path will export them to that path
           distributions applies to both the simulated 1999 population and the NHANES 2017 comparison population
           Returns {"simSummary", "nhanesSummary", "cvRates", "dementiaIncidence"}: the last-wave
           distribution summary of the advanced population and of the NHANES comparison population,
           the standardized CV rates and the dementia incidence, for print_over_time_with_burke2024.'''
        nYears = 18
        popSize = 100000
        #like the Burke2024 cohort, the advanced population excludes people with a history of
        #stroke or MI; a person-level filter, since the seeded priorToSim outcomes only exist on
        #a constructed Person. No prior CV is equivalent: the MI prevalence partition gives every
        #seeded-CV person either a stroke or an MI
        pfSim = PersonFilterFactory.get_person_filter(["adult"])
        pfSim.add_filter(filterType="person",
                         filterName="noPriorCV",
                         filterFunction = lambda x: not x.has_outcome_prior_to_simulation(OutcomeType.CARDIOVASCULAR))
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=1999, personFilters=pfSim, nhanesWeights=True, distributions=distributions)
        pop.advance_parallel(nYears, None, nWorkers)
        #the comparison population is restricted to people who were plausibly in the US in 1999
        #(the advanced cohort cannot gain post-1999 immigrants) and to ages the cohort can reach
        pf = PersonFilterFactory.get_person_filter(["usBornOrInUs15PlusYears"])
        pf.add_filter(filterType="df",
                      filterName="lowAge",
                      filterFunction = lambda x: x[DynamicRiskFactorsType.AGE.value]>=36)
        nhanesPop = PopulationFactory.get_nhanes_population(n=popSize, year=2017, personFilters=pf, nhanesWeights=True, distributions=distributions)

        print("\nVALIDATION OF VASCULAR RISK FACTORS OVER TIME")
        pop.plot_vascular_rfs_last_wave(nhanesPop, path=path)
        pop.print_lastyear_summary_comparison(nhanesPop)
        print("\nVALIDATION OF CV EVENT INCIDENCE AND MORTALITY")
        cvRates = pop.get_cv_standardized_rates()
        pop.print_cv_standardized_rates(rates=cvRates)
        print("\nVALIDATION OF DEMENTIA INCIDENCE")
        dementiaIncidence = pop.get_outcome_incidence(OutcomeType.DEMENTIA)
        pop.print_outcome_incidence(outcomeType=OutcomeType.DEMENTIA, summary=dementiaIncidence)
        return {"simSummary": pop.get_summary_at_index(-1),
                "nhanesSummary": nhanesPop.get_summary_at_index(-1),
                "cvRates": cvRates,
                "dementiaIncidence": dementiaIncidence}

    @staticmethod
    def nhanes_treatment_effects(sampleSize=2000000, nWorkers=1):
        '''This function creates and advances a control and a treated population in order to estimate the
           BP medication treatment effect on the MI relative risk and the stroke relative risk.
           Returns {bpMedsAdded: {"strokeRR", "miRR", "strokeRRs", "miRRs"}}, the mean relative risk
           of each arm over its simulations and the individual simulation results, for
           print_treatment_effects_with_burke2024.'''
        print("\nVALIDATION OF TREATMENT EFFECTS")
        nYears=5
        nSimulations = 4
        results = dict()
        #NHANES includes children, so the adult filter is what keeps this an adult treatment effect
        pf = PersonFilterFactory.get_person_filter(["adult"])
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
            results[bpMedsAdded] = {"strokeRR": float(np.mean(strokeRRList)),
                                    "miRR": float(np.mean(miRRList)),
                                    "strokeRRs": strokeRRList,
                                    "miRRs": miRRList}
        return results

    # ==========================================================================
    # The NHANES validation results next to the published values of Burke2024
    # ==========================================================================

    @staticmethod
    def _print_side_by_side(title, columns, rows, fmt="{:.1f}"):
        '''Prints one section of the report: a row per quantity, the value Burke2024 publishes for
           it next to the value this run obtained.
           columns: the headers of the columns of this run, one per value of a row
           rows: (quantity, published value, [values of this run]); a published value given as a
                 (low, high) tuple is one the paper publishes as a range
           fmt: the format of every number of the section'''
        print(f"\n{title}")
        print("-"*len(title))
        print(f"{'quantity':<44}{'published':>14}" +
              "".join([f"{column:>14}" for column in columns]))
        for quantity, published, values in rows:
            publishedString = (f"{fmt.format(published[0])}-{fmt.format(published[1])}"
                               if type(published) == tuple else fmt.format(published))
            print(f"{quantity:<44}{publishedString:>14}" +
                  "".join([f"{fmt.format(value):>14}" for value in values]))

    @staticmethod
    def print_baseline_pop_with_burke2024(results):
        '''Prints the baseline populations next to the published NHANES estimates of Table 2.
           results: what nhanes_baseline_pop returns.'''
        for title, published, metrics in [
                ("Baseline population (Table 2, published NHANES 2007-2010)",
                 Burke2024.baseline2007, results["2007"]),
                ("Baseline population with hypertension (Table 2, published NHANES 2013)",
                 Burke2024.baseline2013Hypertension, results["2013Hypertension"])]:
            Validation._print_side_by_side(title, ["simulation"],
                                           [(quantity, value, [metrics[quantity]])
                                            for quantity, value in published.items()])

    @staticmethod
    def print_over_time_with_burke2024(results):
        '''Prints the population advanced for 18 years next to the published NHANES 2017 levels of
           the Fig 2 text and the published event rates of Table 3. The NHANES 2017 column is the
           comparison population that nhanes_over_time built in the same run, which is what the
           simulation is held against in Fig 2.
           results: what nhanes_over_time returns.'''
        simSummary = results["simSummary"]["continuous"]
        nhanesSummary = results["nhanesSummary"]["continuous"]
        Validation._print_side_by_side(
            "Vascular risk factors after 18 years (Fig 2, published NHANES 2017)",
            ["NHANES 2017", "simulation"],
            [(f"{riskFactor} ({stat})", published,
              [nhanesSummary[riskFactor][stat], simSummary[riskFactor][stat]])
             for (riskFactor, stat), published in Burke2024.nhanes2017.items()])

        cvRates = results["cvRates"]
        rates = {"MI incidence, all (per 100,000)": cvRates[OutcomeType.MI]["all"],
                 "MI incidence, white (per 100,000)": cvRates[OutcomeType.MI]["white"],
                 "MI incidence, black (per 100,000)": cvRates[OutcomeType.MI]["black"],
                 "stroke incidence, all (per 100,000)": cvRates[OutcomeType.STROKE]["all"],
                 "stroke incidence, white (per 100,000)": cvRates[OutcomeType.STROKE]["white"],
                 "stroke incidence, black (per 100,000)": cvRates[OutcomeType.STROKE]["black"],
                 "mortality, all (per 100,000)": cvRates[OutcomeType.DEATH]["all"]}
        Validation._print_side_by_side(
            "Event incidence and mortality (Table 3)", ["simulation"],
            [(quantity, published, [rates[quantity]])
             for quantity, published in Burke2024.cvIncidence.items()])

    @staticmethod
    def print_treatment_effects_with_burke2024(results):
        '''Prints the estimated BP medication treatment effects next to the meta-analysis of
           BP-lowering trials that Burke2024 calibrated against. The meta-analysis covers a single
           added medication, so that is the arm printed here; nhanes_treatment_effects prints all
           of the arms it runs.
           results: what nhanes_treatment_effects returns.'''
        if 1 not in results:
            return
        relativeRisks = {"stroke RR, 1 BP medication added": results[1]["strokeRR"],
                         "MI RR, 1 BP medication added": results[1]["miRR"]}
        Validation._print_side_by_side(
            "BP medication treatment effects (trial meta-analysis)", ["simulation"],
            [(quantity, published, [relativeRisks[quantity]])
             for quantity, published in Burke2024.treatmentEffects.items()],
            fmt="{:.2f}")

    @staticmethod
    def nhanes_burke2024_report(baselineResults=None, overTimeResults=None,
                                treatmentEffectsResults=None, path=None, nWorkers=5, distributions=False):
        '''Prints the results of the NHANES validation next to the values published in Burke2024.

           Each argument is what the corresponding nhanes_* function returns, and any that is None
           is obtained by running that function here. So

               Validation.nhanes_burke2024_report()

           runs the entire NHANES validation and reports on it, whereas passing results that are
           already in hand reports on them without running anything again - which is what the
           arguments are for, since these runs take hours.'''
        if baselineResults is None:
            baselineResults = Validation.nhanes_baseline_pop()
        if overTimeResults is None:
            overTimeResults = Validation.nhanes_over_time(nWorkers=nWorkers, path=path, distributions=distributions)
        if treatmentEffectsResults is None:
            treatmentEffectsResults = Validation.nhanes_treatment_effects()

        print("\n\nTHIS RUN AND BURKE2024")
        print("======================")
        print("Burke et al., Development and validation of the Michigan Chronic Disease")
        print("Simulation Model (MICROSIM), PLOS ONE 19(5):e0300005, 2024.")
        print("The published column holds the external values the paper validated MICROSIM")
        print("against: survey-weighted NHANES estimates, population-based incidence studies,")
        print("US population mortality, and a meta-analysis of BP-lowering trials.")
        print("\nThe runs printed here differ from the ones the paper describes: 100,000 people")
        print("rather than 500,000 at baseline and 250,000 over time, 4 simulations rather than")
        print("15 for the treatment effects, and a population advanced over time that excludes")
        print("prior stroke and MI like the paper's but, unlike the paper's, not prior dementia.")

        Validation.print_baseline_pop_with_burke2024(baselineResults)
        Validation.print_over_time_with_burke2024(overTimeResults)
        Validation.print_treatment_effects_with_burke2024(treatmentEffectsResults)

    @staticmethod
    def nhanes(path=None, compare=True, nWorkers=5, distributions=False):
        '''Runs the entire NHANES validation.
           compare=True also prints the results next to the published values of Burke2024, see
           nhanes_burke2024_report.
           distributions applies to the over-time populations, see nhanes_over_time.
           Returns the results of the three validation runs.'''
        baselineResults = Validation.nhanes_baseline_pop()
        overTimeResults = Validation.nhanes_over_time(nWorkers=nWorkers, path=path, distributions=distributions)
        treatmentEffectsResults = Validation.nhanes_treatment_effects()
        if compare:
            Validation.nhanes_burke2024_report(baselineResults, overTimeResults,
                                               treatmentEffectsResults)
        return {"baseline": baselineResults,
                "overTime": overTimeResults,
                "treatmentEffects": treatmentEffectsResults}

    @staticmethod
    def nhanes_prevalence_by_age(outcomeType=OutcomeType.STROKE, popSize=100000, year=2017):
        '''Creates a nationally representative US population (NHANES survey-weighted, adults) and
           prints the baseline prevalence of an outcome by 5-year age group. The population is not
           advanced, so this is the seeded priorToSim prevalence. When Reference.prevalence holds
           rates for the outcome, they are printed next to the simulation: split by gender when the
           reference is gender-stratified, pooled otherwise.
           Returns the prevalence dictionary keyed by age group (by gender then age group when the
           reference is gender-stratified).'''
        print(f"\n{outcomeType.value.upper()} PREVALENCE BY AGE GROUP AT BASELINE (NHANES {year}, survey-weighted)")
        pop = PopulationFactory.get_nhanes_population(n=popSize, year=year, personFilters=None,
                                                      nhanesWeights=True, distributions=False)
        reference = Reference.prevalence.get(outcomeType.value)
        if reference is not None and "male" in reference:
            prevalence = pop.get_prevalence_by_age(outcomeType, groups=True, byGender=True)
            for gender in prevalence.keys():
                print(f"{gender}")
                print(f"{'age group':>12}{'simulation':>14}{'reference':>14}")
                for ageGroup, value in prevalence[gender].items():
                    ref = reference[gender].get(ageGroup)
                    refString = f"{ref:>14.4f}" if ref is not None else f"{'':>14}"
                    print(f"{ageGroup:>12}{value:>14.4f}{refString}")
        else:
            prevalence = pop.get_prevalence_by_age(outcomeType, groups=True)
            print(f"{'age group':>12}{'simulation':>14}{'reference':>14}")
            for ageGroup, value in prevalence.items():
                ref = reference.get(ageGroup) if reference is not None else None
                refString = f"{ref:>14.4f}" if ref is not None else f"{'':>14}"
                print(f"{ageGroup:>12}{value:>14.4f}{refString}")
        return prevalence

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

 

