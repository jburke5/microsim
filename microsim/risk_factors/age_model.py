class AgeModel:
    def __init__(self):
        pass

    def estimate_next_risk(self, person):
        return person._age[-1]+1
