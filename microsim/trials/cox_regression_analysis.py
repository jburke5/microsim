from lifelines import CoxPHFitter
from numpy.linalg import LinAlgError
from lifelines.exceptions import ConvergenceError
import numpy as np
import pandas as pd
from microsim.trials.regression_analysis import RegressionAnalysis

class CoxRegressionAnalysis(RegressionAnalysis):
    columns = ("coef", "se", "pValue", "intercept") #intercept is always None

    def __init__(self):
        self.cph = CoxPHFitter()
    
    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        df = self.get_trial_outcome_df(trial, assessmentFunctionDict, assessmentAnalysis)
        #the analysis adjusts for all block factors, but randomization blocks only on blockFactors[0]
        blockFactors = trial.trialDescription.blockFactors
        df = df.loc[:,["outcome", "outcomeTime", "treatment", *blockFactors]]
        #categorical block factors are dummy-encoded, otherwise lifelines would fit them as a single linear term
        categoricalFactors = [bf for bf in blockFactors if self.is_categorical(bf)]
        if len(categoricalFactors)>0:
            df = pd.get_dummies(df, columns=categoricalFactors, drop_first=True, dtype=float)
        self.cph = CoxPHFitter() #fresh fitter, so no state from a previous assessment is carried over
        try:
            self.cph.fit(df, duration_col=f"outcomeTime", event_col="outcome")
            return self.cph.params_['treatment'], self.cph.standard_errors_['treatment'], self.cph.summary.loc['treatment', 'p'], None
        except (LinAlgError, ConvergenceError):
            return np.nan, np.nan, np.nan, None #cox has no intercept, the 4th element is always None

