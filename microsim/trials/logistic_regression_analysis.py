import statsmodels.formula.api as smf
from microsim.trials.regression_analysis import RegressionAnalysis

class LogisticRegressionAnalysis(RegressionAnalysis):
    def __init__(self, outcomeAssessor, nameStem='logisticRegression-'):
        super().__init__(outcomeAssessor, nameStem)

    def analyze(self, treatedPop, untreatedPop):
        reg = smf.logit("outcome ~ treatment", data=self.get_dataframe(treatedPop, untreatedPop)).fit(disp=False, method_kwargs={"warn_convergence": False})
        return reg.params['treatment'], reg.bse['treatment'], reg.pvalues['treatment']


