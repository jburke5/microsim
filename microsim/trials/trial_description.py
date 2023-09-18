import numpy as np

class TrialDescription:
    def __init__(self, sampleSizes, durations, inclusionFilters, exclusionFilters, analyses, treatment,
                randomizationSchema):

        self.sampleSizes = sampleSizes
        self.durations = durations
        self.inclusionFilters = inclusionFilters
        self.exclusionFilters = exclusionFilters
        self.randomizationSchema = randomizationSchema
        self.treatment = treatment
        self.analyses = analyses
