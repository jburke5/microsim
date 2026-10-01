import pandas as pd

from microsim.risk_factors.risk_factor import CategoricalRiskFactorsType

class RegressionAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def is_categorical(blockFactor):
        return blockFactor in [rf.value for rf in CategoricalRiskFactorsType]

    def get_trial_outcome_df(self, trial, assessmentFunctionDict, assessmentAnalysis):
        treatment = [1]*trial.treatedPop._n+[0]*trial.controlPop._n
        dfDict=dict()
        dfDict["treatment"]=treatment
        assessmentFunction = assessmentFunctionDict["outcome"]
        dfDict["outcome"] = assessmentFunction(trial.treatedPop) + assessmentFunction(trial.controlPop)
        if assessmentAnalysis=="logistic":
            dfDict["outcome"] = [int(x) for x in dfDict["outcome"]]
        elif assessmentAnalysis=="cox":
            assessmentFunction = assessmentFunctionDict["time"]
            dfDict["outcomeTime"] = assessmentFunction(trial.treatedPop) + assessmentFunction(trial.controlPop)
        #the analyses adjust for all block factors, but note that randomization blocks only on blockFactors[0]
        for blockFactor in trial.trialDescription.blockFactors:
            blockValues = trial.treatedPop.get_attr(blockFactor) + trial.controlPop.get_attr(blockFactor)
            if any(isinstance(v, list) for v in blockValues):
                raise RuntimeError(f"Block factor {blockFactor} must be a static attribute, not a dynamic risk factor.")
            dfDict[blockFactor] = blockValues
        return pd.DataFrame(dfDict)




