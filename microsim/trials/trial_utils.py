import numpy as np

def get_analysis_name(analysis, duration, sampleSize):
    return f"{analysis.name}-{str(duration)}Years-{sampleSize}"

#this function can be generalized, for more distributions
def randomizationSchema(x):
    return (np.random.uniform() < 0.5) #assumes a uniform distribution

