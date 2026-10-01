import sys
import warnings
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tools.sm_exceptions
from numpy.linalg import LinAlgError
from microsim.trials.regression_analysis import RegressionAnalysis

class LogisticRegressionAnalysis(RegressionAnalysis):
    columns = ("coef", "se", "pValue", "intercept")

    def __init__(self):
        pass

    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        df = self.get_trial_outcome_df(trial, assessmentFunctionDict, assessmentAnalysis)
        blockFactors = trial.trialDescription.blockFactors
        formula = f"outcome ~ treatment"
        for blockFactor in blockFactors:
            #categorical block factors are dummy-encoded by patsy, otherwise they would be fit as a single linear term
            formula += f" + C({blockFactor})" if self.is_categorical(blockFactor) else f" + {blockFactor}"
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=statsmodels.tools.sm_exceptions.PerfectSeparationWarning)
                reg = smf.logit(formula, df).fit(disp=False)
            return reg.params['treatment'], reg.bse['treatment'], reg.pvalues['treatment'], reg.params['Intercept']
        except (LinAlgError,
                statsmodels.tools.sm_exceptions.PerfectSeparationError, #some statsmodels code paths raise the error directly
                statsmodels.tools.sm_exceptions.PerfectSeparationWarning):
            print("Logistic regression failed (perfect separation/singular matrix), returning NaN.", file=sys.stderr)
            return np.nan, np.nan, np.nan, np.nan



