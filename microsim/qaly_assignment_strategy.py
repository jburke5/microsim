


class QALYAssignmentStrategy:
    def __init__(self):
        pass
    
    def get_next_qaly(self, person):
        return self.get_base_qaly_for_age(person)
    
    # simple age-based approximation that after age 70, you lose about 0.01 QALYs per year
    def get_base_qaly_for_age(self, person):
        yearsOver70 = person._age[-1] - 70 if person._age[-1] > 70 else 0
        return 1 - yearsOver70 * 0.01
