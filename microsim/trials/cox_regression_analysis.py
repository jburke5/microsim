from lifelines import CoxPHFitter
from numpy.linalg import LinAlgError
from lifelines.exceptions import ConvergenceError
import numpy as np
from microsim.trials.regression_analysis import RegressionAnalysis

class CoxRegressionAnalysis(RegressionAnalysis):
    def __init__(self, outcomeAssessor, nameStem='coxRegression-'):
        super().__init__(outcomeAssessor, nameStem)

    def analyze(self, treatedPop, untreatedPop):
        data=self.get_dataframe(treatedPop,untreatedPop,includeTime=True)
        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col='time', event_col='outcome')

            return cph.params_['treatment'], None, cph.standard_errors_['treatment'], cph.summary.loc['treatment', 'p'], self.get_means(data)
        except (LinAlgError, ConvergenceError):
            return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan)


