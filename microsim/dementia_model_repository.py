from microsim.dementia_model import DementiaModel
#from microsim.outcome import OutcomeType
from microsim.modality import Modality

class DementiaModelRepository:
    def __init__(self, wmhSpecific=True):
        #if we want the dementia model to return an average risk for the kaiser population, independently of their WMH outcome, then
        #we need to both adjust the intercept and make sure the model does not utilize the WMH outcome
        #I think we could also implement this in the same way the non cv model repository is setup (push the intercept changes to the model initialization)
        if wmhSpecific:
            brainScanModel =  DementiaModel(linearTerm=1.33371239e-05 + 1.0e-04, quadraticTerm=5.64485841e-05 + 5.14857143e-05, wmhSpecific=wmhSpecific) 
        else:
            brainScanModel =  DementiaModel(linearTerm=1.33371239e-05 + 0.0001, quadraticTerm=5.64485841e-05 + 0.0002575, wmhSpecific=wmhSpecific)
        self._models = {"NHANES": DementiaModel(), #default linear and quadratic terms for NHANES
                        #this is what I had from the initial optimization, not sure how well it will work now that I modified death rates
                        #"brainScan": DementiaModel(linearTerm=3.05555556e-04, quadraticTerm=2.40000000e-06)} #had a brain scan
                        #"brainScan": DementiaModel(linearTerm=1.33371239e-05 + 2.42857143e-04, quadraticTerm=5.64485841e-05 + 2.59428571e-05)} #had a brain scan
                        "brainScan": brainScanModel}
        
    def select_outcome_model_for_person(self, person):
        '''Use modality to select the appropriate dementia model because not all Kaiser population members have silent
        cerebrovascular disease, but all of them had some kind of imaging done on their brains.
        So, it is the imaging that determined participation in this group, not silent cerebrovascular disease.'''
        brainScan = not person._modality == Modality.NO.value #if person had a brain scan or not
        return self._models["brainScan"] if brainScan else self._models["NHANES"]
        
