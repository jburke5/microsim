import statsmodels.formula.api as smf
from microsim.trials.regression_analysis import RegressionAnalysis

class LogisticRegressionAnalysis(RegressionAnalysis):
    def __init__(self, outcomeAssessor, nameStem='logisticRegression-'):
        super().__init__(outcomeAssessor, nameStem)

    def analyze(self, treatedPop, untreatedPop):
        data=self.get_dataframe(treatedPop,untreatedPop)
        reg = smf.logit("outcome ~ treatment", data).fit(disp=False)
        return reg.params['treatment'], reg.params['Intercept'], reg.bse['treatment'], reg.pvalues['treatment'], self.get_means(data)


