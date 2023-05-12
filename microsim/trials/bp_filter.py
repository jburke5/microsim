class BPFilter:
    # bpThresholds takes a dict where the key is sbp or dbp, and the value is a float 

    def __init__(self, bpThresholds):
        self._bpThresholds = bpThresholds
    
    #exceedsThresholds function is using a logical AND for all blood pressure types
    def exceedsThresholds(self, person):
        exceedsThresholds = True
        for bpType, bpThreshold in self._bpThresholds.items():
            bp = getattr(person, bpType)[-1]
            if bp < bpThreshold:
                exceedsThresholds = False
                break

        return exceedsThresholds

    def doesNotExceedThresholds(self, person):
        return (not self.exceedsThresholds(person))
