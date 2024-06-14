import statsmodels.formula.api as smf
from microsim.trials.regression_analysis import RegressionAnalysis

class LogisticRegressionAnalysis(RegressionAnalysis):
    def __init__(self):
        pass
    
    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        df = self.get_trial_outcome_df(trial, assessmentFunctionDict, assessmentAnalysis)
        blockFactors = trial.trialDescription.blockFactors
        formula = f"outcome ~ treatment"
        for blockFactor in blockFactors:
            formula += f" + {blockFactor}"
        reg = smf.logit(formula, df).fit(disp=False)
        return reg.params['treatment'], reg.params['Intercept'], reg.bse['treatment'], reg.pvalues['treatment']



