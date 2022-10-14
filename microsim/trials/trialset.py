from microsim.trials.trial import Trial
from microsim.trials.trial_utils import get_analysis_name
import pandas as pd

class Trialset:

    def __init__(self, trialDescription, pop, trialCount):
        self.trials = []
        self.trialDescription = trialDescription
        for i in range(0, trialCount):
            self.trials.append(Trial(trialDescription, pop))

    def run(self):
        for trial in self.trials:
            trial.run()
        self.resultsDict = self.build_all_results_df()

    def get_all_results_dict(self):
        return self.resultsDict
    
    def get_results_for_analysis(self, analysis):
        return self.resultsDict[analysis]

    def build_all_results_df(self):
        results = {}
        for analysis in self.trialDescription.analyses:
            for duration in self.trialDescription.durations:
                resultsForAnalysis = []
                for trial in self.trials:
                    resultsForAnalysis.append(trial.analyticResults[get_analysis_name(analysis, duration)])
                results[get_analysis_name(analysis, duration)] = pd.DataFrame(resultsForAnalysis)
        return results

def get_analysis_name(analysis, duration):
    return f"{analysis.name}-{str(duration)}"