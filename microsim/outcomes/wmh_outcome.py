from microsim.outcomes.outcome import Outcome, OutcomeType

# Map to be used for the classification of person objects regarding the WMH outcome.
# This serves as the categorical variable to be used later on with regression, as a covariate.
# This is another representation of the map in table form:
#            sbi=False    sbi=True
# wmh=False       0            1
# wmh=True        2            3
scdGroupMap = [
    [0, 1],
    [2, 3],
]  # no sbi & no wmh -> 0, sbi only -> 1, wmh only -> 2, both sbi & wmh -> 3


class WMHOutcome(Outcome):
    phenotypeItems = ["sbi", "wmh", "wmhSeverityUnknown", "wmhSeverity"]

    def __init__(self, fatal, sbi, wmh, wmhSeverityUnknown, wmhSeverity, priorToSim=False):
        self.fatal = fatal
        self.priorToSim = priorToSim
        super().__init__(OutcomeType.WMH, self.fatal, self.priorToSim)
        self.sbi = sbi
        self.wmh = wmh
        self.wmhSeverityUnknown = wmhSeverityUnknown
        self.wmhSeverity = wmhSeverity

    def __repr__(self):
        return (
            f"WMH Outcome: {self.type}, fatal: {self.fatal}, sbi: {self.sbi}, wmh: {self.wmh},\n"
            f"                   wmhSeverityUnknown: {self.wmhSeverityUnknown}, "
            f"wmhSeverity: {self.wmhSeverity}"
        )
