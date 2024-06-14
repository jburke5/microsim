import pandas as pd

class RegressionAnalysis:
    def __init__(self):
        pass
    
    def get_trial_outcome_df(self, trial, assessmentFunctionDict, assessmentAnalysis):
        treatment = [1]*trial.treatedPop._n+[0]*trial.controlPop._n
        dfDict=dict()
        dfDict["treatment"]=treatment
        assessmentFunction = assessmentFunctionDict["outcome"]
        dfDict["outcome"] = list(map(assessmentFunction, [trial.treatedPop]))[0] + list(map(assessmentFunction, [trial.controlPop]))[0]
        if assessmentAnalysis=="logistic":
            dfDict["outcome"] = [int(x) for x in dfDict["outcome"]]
        elif assessmentAnalysis=="cox":
            assessmentFunction = assessmentFunctionDict["time"]
            dfDict["outcomeTime"] = list(map(assessmentFunction, [trial.treatedPop]))[0] + list(map(assessmentFunction, [trial.controlPop]))[0]
        if len(trial.trialDescription.blockFactors)>0:
            blockFactor = trial.trialDescription.blockFactors[0]
            dfDict[blockFactor]=trial.treatedPop.get_attr(blockFactor) + trial.controlPop.get_attr(blockFactor)      
        return pd.DataFrame(dfDict) 




