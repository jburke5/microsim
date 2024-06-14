from lifelines import CoxPHFitter
from numpy.linalg import LinAlgError
from lifelines.exceptions import ConvergenceError
import numpy as np
from microsim.trials.regression_analysis import RegressionAnalysis

class CoxRegressionAnalysis(RegressionAnalysis):
    def __init__(self):
        self.cph = CoxPHFitter()
    
    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        df = self.get_trial_outcome_df(trial, assessmentFunctionDict, assessmentAnalysis)
        blockFactors = trial.trialDescription.blockFactors
        try:
            self.cph.fit(df.loc[:,["outcome", "outcomeTime", "treatment", *blockFactors]], duration_col=f"outcomeTime", event_col="outcome")
            return self.cph.params_['treatment'], None, self.cph.standard_errors_['treatment'], self.cph.summary.loc['treatment', 'p']
        except (LinAlgError, ConvergenceError):
            return np.nan, np.nan, np.nan, np.nan

