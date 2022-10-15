import pandas as pd
import numpy as np

class RegressionAnalysis:
    def __init__(self, outcomeAssessor, nameStem):
        self.outcomeAssessor = outcomeAssessor
        self.name = nameStem + outcomeAssessor.get_name()

    def get_dataframe(self, treatedPopList, untreatedPopList):
        treatedOutcomes = [self.outcomeAssessor.get_outcome(person) for i, person in enumerate(treatedPopList)]
        untreatedOutcomes = [self.outcomeAssessor.get_outcome(person) for i, person in enumerate(untreatedPopList)]

        treatedOutcomes.extend(untreatedOutcomes)
        analysisDF = pd.DataFrame({'treatment' : np.append(np.ones(len(treatedPopList)), np.zeros(len(untreatedPopList))),
                    'outcome' : treatedOutcomes})
        analysisDF.outcome = analysisDF.outcome.astype('int')
        return analysisDF
    




