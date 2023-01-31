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
        analysisDF = pd.DataFrame({'treatment' : np.append(np.ones(len(treatedPopList),dtype=int), np.zeros(len(untreatedPopList),dtype=int)),
                    'outcome' : treatedOutcomes})
        analysisDF.outcome = analysisDF.outcome.astype('int')
        return analysisDF
    
    def get_means(self, analysisDF):
        #column names and flag values in this method are based on get_dataframe method above
        analysisDFTreated = analysisDF.loc[analysisDF["treatment"]==1]
        analysisDFUntreated = analysisDF.loc[analysisDF["treatment"]==0]
        
        #for logistic regression: returns proportions (# of outcomes)/(# of people in group) in control and treated groups
        #for linear regression: returns attribute mean in control and treated groups
        #note: I can use mean for both because LogisticRegressionAnalysis uses, exclusively for now I think,
        #OutcomeAssessor which returns a value of 1 for True and 0 for False and this allows an easy calculation of proportions
        
        return analysisDFUntreated["outcome"].mean(), analysisDFTreated["outcome"].mean()




