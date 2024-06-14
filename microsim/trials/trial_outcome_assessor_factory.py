from microsim.trials.trial_outcome_assessor import TrialOutcomeAssessor
from microsim.outcome import OutcomeType

class TrialOutcomeAssessorFactory:

    @staticmethod
    def get_trial_outcome_assessor(addCommonAssessments=True):
        '''This function adds some trial outcome assessments that are likely to be interesting from a trial.
        It also serves as an example of how trial outcome assessments can be added.'''
        toa = TrialOutcomeAssessor()
        if addCommonAssessments:
            toa.add_outcome_assessment("death", 
                                       {"outcome": lambda x: x.has_outcome(OutcomeType.DEATH)}, 
                                        "logistic")
            toa.add_outcome_assessment("anyEvent", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE,
                                                                  OutcomeType.DEMENTIA, OutcomeType.CI])}, 
                                        "logistic")
            toa.add_outcome_assessment("vascularEventOrDeath", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE])}, 
                                        "logistic")
            toa.add_outcome_assessment("vascularEvent", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.MI, OutcomeType.STROKE])}, 
                                        "logistic")
            toa.add_outcome_assessment("qalys", 
                                       {"outcome": lambda x: x.get_outcome_item_sum(OutcomeType.QUALITYADJUSTED_LIFE_YEARS, "qaly")}, 
                                        "linear")
            toa.add_outcome_assessment("meanGCP", 
                                       {"outcome": lambda x: x.get_outcome_item_mean(OutcomeType.COGNITION, "gcp")}, 
                                        "linear")
            toa.add_outcome_assessment("lastGCP", 
                                       {"outcome": lambda x: x.get_outcome_item_last(OutcomeType.COGNITION, "gcp")}, 
                                        "linear")
            toa.add_outcome_assessment("cogEvent", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.CI, OutcomeType.DEMENTIA])}, 
                                        "logistic")
            toa.add_outcome_assessment("deathCox", 
                                       {"outcome": lambda x: x.has_outcome(OutcomeType.DEATH),
                                        "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.DEATH])},
                                        "cox")
            toa.add_outcome_assessment("cogEventCox", 
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.CI, OutcomeType.DEMENTIA]),
                                        "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.CI, OutcomeType.DEMENTIA])},
                                        "cox")
            toa.add_outcome_assessment("vascularEventOrDeathCox",
                                       {"outcome": lambda x: x.has_any_outcome([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE]),
                                        "time": lambda x: x.get_min_wave_of_first_outcomes_or_last_wave([OutcomeType.DEATH, OutcomeType.MI, OutcomeType.STROKE])},
                                        "cox")
            toa.add_outcome_assessment("strokeRR",
                                       {"outcome": lambda x: x.get_outcome_risk(OutcomeType.STROKE)},
                                        "relRisk")
            toa.add_outcome_assessment("miRR",
                                       {"outcome": lambda x: x.get_outcome_risk(OutcomeType.MI)},
                                        "relRisk")
        return toa
