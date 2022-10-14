import numpy as np

class TrialDescription:
    def __init__(self, sampleSize, durations, inclusionFilter, exclusionFilter, analyses, treatment,
                randomizationSchema=lambda x : np.random.uniform() < 0.5):

        self.sampleSize = sampleSize
        self.durations = durations
        self.inclusionFilter = inclusionFilter
        self.exclusionFilter = exclusionFilter
        self.randomizationSchema = randomizationSchema
        self.treatment = treatment
        self.analyses = analyses
