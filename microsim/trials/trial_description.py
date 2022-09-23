import numpy as np

class TrialDescription:
    def __init__(self, sampleSize, duration, inclusionFilter, exclusionFilter, outcomes, 
        randomizationSchema=lambda x : np.random.uniform() < 0.5):

        self.sampleSize = sampleSize
        self.duration = duration 
        self.inclusionFilter = inclusionFilter
        self.exclusionFilter = exclusionFilter
        self.randomizationSchema = randomizationSchema
