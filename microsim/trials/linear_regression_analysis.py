import numpy as np
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError
from microsim.trials.regression_analysis import RegressionAnalysis

class LinearRegressionAnalysis(RegressionAnalysis):
    columns = ("coef", "se", "pValue", "intercept")

    def __init__(self):
        pass
    
    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        df = self.get_trial_outcome_df(trial, assessmentFunctionDict, assessmentAnalysis)
        #the analysis adjusts for all block factors, but randomization blocks only on blockFactors[0]
        blockFactors = trial.trialDescription.blockFactors
        formula = f"outcome ~ treatment"
        for blockFactor in blockFactors:
            #categorical block factors are dummy-encoded by patsy, otherwise they would be fit as a single linear term
            formula += f" + C({blockFactor})" if self.is_categorical(blockFactor) else f" + {blockFactor}"
        try:
            reg = smf.ols(formula, df).fit()
            return reg.params['treatment'], reg.bse['treatment'], reg.pvalues['treatment'], reg.params['Intercept']
        except (LinAlgError, ValueError):
            return np.nan, np.nan, np.nan, np.nan


