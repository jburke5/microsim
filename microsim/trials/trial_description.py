

class TrialDescription:
    def __init__(self, sampleSize, duration, inclusionFilter, exclusionFilter, outcomes, 
        randomizationSchema):

        self.sampleSize = sampleSize
        self.duration = duration 
        self.inclusionFilter = inclusionFilter
        self.exclusionFilter = exclusionFilter
        self.randomizationSchema = randomizationSchema
