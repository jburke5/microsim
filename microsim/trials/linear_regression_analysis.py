import statsmodels.formula.api as smf
from microsim.trials.regression_analysis import RegressionAnalysis

class LinearRegressionAnalysis(RegressionAnalysis):
    def __init__(self, outcomeAssessor, nameStem='linearRegression-'):
        super().__init__(outcomeAssessor, nameStem)

    def analyze(self, treatedPop, untreatedPop):
        reg = smf.ols("outcome ~ treatment", data=self.get_dataframe(treatedPop, untreatedPop)).fit(disp=False)
        return reg.params['treatment'], reg.bse['treatment'], reg.pvalues['treatment']


