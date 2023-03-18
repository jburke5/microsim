import numpy as np

#sampleSizeIndex is needed due to the trial method analyzeSmallerTrials, without it trial.analyticResults would overwrite smaller sample results
def get_analysis_name(analysis, duration, sampleSize, sampleSizeIndex=0):
    return f"{analysis.name}-{str(duration)}Years-{sampleSize}-{sampleSizeIndex}"

#this function can be generalized, for more distributions
def randomizationSchema(x, rng=None):
    rng = np.random.default_rng(rng)
    return (rng.uniform() < 0.5) #assumes a uniform distribution

