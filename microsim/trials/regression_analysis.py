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
    
    def get_absolute_effect_size(self, analysisDF):
        #column names and flag values in this method are based on get_dataframe method above
        analysisDFTreatment = analysisDF.loc[analysisDF["treatment"]==1]
        analysisDFControl = analysisDF.loc[analysisDF["treatment"]==0]
        
        #for logistic regression: returns difference of proportions (# of outcomes)/(# of people in group)
        #for linear regression: returns difference of attribute means
        #note: I can use mean for both because LogisticRegressionAnalysis uses, exclusively for now I think,
        #OutcomeAssessor which returns a value of 1 for True and 0 for False and this allows an easy
        #calculation of proportions
        absoluteEffectSize =  analysisDFTreatment["outcome"].mean() - analysisDFControl["outcome"].mean()
        
        return absoluteEffectSize




