class AgeModel:
    def __init__(self):
        pass
 
    def estimate_next_risk(self, person):
        return person._age[-1]+1
   
    def get_next_risk_factor(self, person):
        return person._age[-1]+1
