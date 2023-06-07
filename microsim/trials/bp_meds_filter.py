class BPMedsFilter:
    # bpMedsThresholds takes a dict where the key is bp med type, and the value is a float 

    def __init__(self, bpMedsThresholds):
        self._bpMedsThresholds = bpMedsThresholds
    
    #exceedsThresholds function is using a logical AND for all blood pressure med types
    def exceedsThresholds(self, person):
        exceedsThresholds = True
        for bpMedType, bpMedThreshold in self._bpMedsThresholds.items():
            bpMedCount = getattr(person, bpMedType)[-1]
            if bpMedCount <= bpMedThreshold:
                exceedsThresholds = False
                break

        return exceedsThresholds

    def doesNotExceedThresholds(self, person):
        return (not self.exceedsThresholds(person))
