from microsim.trials.trial import Trial
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
            trial.analyze()
        self.resultsDict = self.build_all_results_df()

    def get_all_results_dict(self):
        return self.resultsDict
    
    def get_results_for_analysis(self, analysis):
        return self.resultsDict[analysis]

    def build_all_results_df(self):
        results = {}
        for analysis in self.trialDescription.analyses:
            resultsForAnalysis = []
            for trial in self.trials:
                resultsForAnalysis.append(trial.analyticResults[analysis])
            results[analysis] = pd.DataFrame(resultsForAnalysis)
        return results
            