import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

class LogisticRegressionAnalysis:
    def __init__(self, outcomeAssessor, nameStem='logisticRegression'):
        self.outcomeAssessor = outcomeAssessor
        self.name = nameStem + outcomeAssessor.get_name()

    def analyze(self, treatedPop, untreatedPop):
        treatedOutcomes = [self.outcomeAssessor.get_outcome(person) for i, person in treatedPop._people.iteritems()]
        untreatedOutcomes = [self.outcomeAssessor.get_outcome(person) for i, person in untreatedPop._people.iteritems()]

        treatedOutcomes.extend(untreatedOutcomes)
        analysisDF = pd.DataFrame({'treatment' : np.append(np.ones(len(treatedPop._people)), np.zeros(len(untreatedPop._people))),
                    'outcome' : treatedOutcomes})
        analysisDF.outcome = analysisDF.outcome.astype('int')
        reg = smf.logit("outcome ~ treatment", data=analysisDF).fit()
        return reg.params['treatment'], reg.bse['treatment'], reg.pvalues['treatment']


