import statsmodels.formula.api as smf
from microsim.trials.regression_analysis import RegressionAnalysis

class LinearRegressionAnalysis(RegressionAnalysis):
    def __init__(self, outcomeAssessor, nameStem='linearRegression-'):
        super().__init__(outcomeAssessor, nameStem)

    def analyze(self, treatedPop, untreatedPop):
        data=self.get_dataframe(treatedPop, untreatedPop)
        reg = smf.ols("outcome ~ treatment", data).fit(disp=False, method_kwargs={"warn_convergence": False})
        return reg.params['treatment'], reg.params['Intercept'], reg.bse['treatment'], reg.pvalues['treatment'], self.get_means(data)


