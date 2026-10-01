from microsim.outcomes.outcome import Outcome, OutcomeType

# generlized logistic function mapping GCP to MMSE in combined cohrot data
MMSE_CEILING = 30  # ceiling effect
MMSE_LOGISTIC_OFFSET = 0.9924
MMSE_LOGISTIC_GCP_SLOPE = 0.0795
MMSE_LOGISTIC_SHAPE = 0.1786

# population GCP standard deviation (from 300,000 NHANES population, not advanced) and the half-SD
# factor used as the CI threshold
GCP_POPULATION_SD = 10.3099
CI_GCP_CHANGE_SD_FACTOR = 0.5

# linear regression for mean GCP by age and year in simulation (developed on a simulated NHANES
# population),
# the constant standard deviation of that mean GCP regression, and the SD factor used as the MCI
# threshold
GCP_MEAN_INTERCEPT = 72.3182
GCP_MEAN_AGE_COEFFICIENT = -0.2945
GCP_MEAN_YEARS_IN_SIM_COEFFICIENT = -0.5884
GCP_MEAN_SD = 9.05
MCI_GCP_SD_FACTOR = 1.5


class CognitionOutcome(Outcome):
    phenotypeItems = ["gcp"]

    def __init__(self, fatal, priorToSim, gcp):
        self.fatal = fatal
        self.priorToSim = priorToSim
        super().__init__(OutcomeType.COGNITION, self.fatal, self.priorToSim)
        self.gcp = gcp
