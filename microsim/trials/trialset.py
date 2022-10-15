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
        trialCount = 0
        for trial in self.trials:
            if trialCount % 10 == 0:
                print(f"#################\n#################    Trial Count: {trialCount}")
            trial.run()
            trialCount+=1
        self.resultsDict = self.build_all_results_df()

    def get_all_results_dict(self):
        return self.resultsDict
    
    def get_results_for_analysis(self, analysis):
        return self.resultsDict[analysis]

    def build_all_results_df(self):
        results = {}
        for analysis in self.trialDescription.analyses:
            for duration in self.trialDescription.durations:
                for sampleSize in self.trialDescription.sampleSizes:
                    resultsForAnalysis = []
                    for trial in self.trials:
                        resultsForAnalysis.append(trial.analyticResults[get_analysis_name(analysis, duration, sampleSize)])
                    results[get_analysis_name(analysis, duration, sampleSize)] = pd.DataFrame(resultsForAnalysis)
        return results
