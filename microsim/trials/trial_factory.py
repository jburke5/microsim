from microsim.trials.trial import Trial
from microsim.trials.trial_description import NhanesTrialDescription, KaiserTrialDescription
from microsim.trials.trial_outcome_assessor_factory import TrialOutcomeAssessorFactory
from microsim.trials.trial_type import TrialType


class TrialFactory:
    '''One-shot helpers for building, running, and analyzing a Trial in a single call.

    Each helper builds the appropriate population-specific TrialDescription, instantiates
    a Trial (which constructs its treated and control populations), runs both arms, and
    analyzes them with a TrialOutcomeAssessor. The completed Trial is returned so the
    caller can inspect trial.results, print the formatted summary, or export them with
    trial.export_results(path). exportPath=None (default) writes nothing.

    If assessor is None, the default assessor from TrialOutcomeAssessorFactory is used.'''

    @staticmethod
    def run_nhanes(sampleSize,
                   duration,
                   treatmentStrategies=None,
                   trialType=TrialType.COMPLETELY_RANDOMIZED,
                   blockFactors=(),
                   nWorkers=1,
                   personFilters=None,
                   year=1999,
                   nhanesWeights=True,
                   distributions=False,
                   prevalenceRiskScaling=None,
                   assessor=None,
                   notify=True,
                   exportPath=None):
        '''Build, run, and analyze an NHANES-based Trial in a single call.

        Example:
            trial = TrialFactory.run_nhanes(sampleSize=1000, duration=5,
                                            treatmentStrategies="1bpMedsAdded",
                                            exportPath="trial-results.csv") #optional
            print(trial)
        '''
        description = NhanesTrialDescription(trialType=trialType,
                                             blockFactors=list(blockFactors),
                                             sampleSize=sampleSize,
                                             duration=duration,
                                             treatmentStrategies=treatmentStrategies,
                                             nWorkers=nWorkers,
                                             personFilters=personFilters,
                                             year=year,
                                             nhanesWeights=nhanesWeights,
                                             distributions=distributions,
                                             prevalenceRiskScaling=prevalenceRiskScaling)
        return TrialFactory._run(description, assessor, notify, exportPath)

    @staticmethod
    def run_kaiser(sampleSize,
                   duration,
                   treatmentStrategies=None,
                   trialType=TrialType.COMPLETELY_RANDOMIZED,
                   blockFactors=(),
                   nWorkers=1,
                   personFilters=None,
                   wmhSpecific=True,
                   riskScaling=None,
                   assessor=None,
                   notify=True,
                   exportPath=None):
        '''Build, run, and analyze a Kaiser-based Trial in a single call.

        Example:
            trial = TrialFactory.run_kaiser(sampleSize=1000, duration=5,
                                            treatmentStrategies="1bpMedsAdded",
                                            exportPath="trial-results.csv") #optional
            print(trial)
        '''
        description = KaiserTrialDescription(trialType=trialType,
                                             blockFactors=list(blockFactors),
                                             sampleSize=sampleSize,
                                             duration=duration,
                                             treatmentStrategies=treatmentStrategies,
                                             nWorkers=nWorkers,
                                             personFilters=personFilters,
                                             wmhSpecific=wmhSpecific,
                                             riskScaling=riskScaling)
        return TrialFactory._run(description, assessor, notify, exportPath)

    @staticmethod
    def _run(description, assessor, notify, exportPath):
        if assessor is None:
            assessor = TrialOutcomeAssessorFactory.get_trial_outcome_assessor()
        trial = Trial(description)
        trial.run_analyze(assessor, notify=notify, exportPath=exportPath)
        return trial
