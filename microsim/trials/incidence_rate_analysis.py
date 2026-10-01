class IncidenceRateAnalysis:
    """Analysis class for calculating outcome incidence rates per 1000 person-years.

    This analysis computes incidence rates for both treated and control trial arms,
    enabling comparison of event rates adjusted for time at risk.
    """
    columns = ("treatedRatePer1000PY", "controlRatePer1000PY")

    def __init__(self):
        pass

    def analyze(self, trial, assessmentFunctionDict, assessmentAnalysis):
        """Calculate incidence rates per 1000 person-years for treated and control arms.

        Args:
            trial: Trial instance with treatedPop and controlPop
            assessmentFunctionDict: Dictionary with one required key:
                - "eventAndTime": function(population) -> list of (event, personYears) pairs,
                  one pair per person, so the numerator and denominator cannot be mismatched
            assessmentAnalysis: string identifier for this analysis type

        Returns:
            tuple: (treated_rate, control_rate) - both as events per 1000 person-years,
            nan for an arm with no person-years
        """
        eventAndTimeFunc = assessmentFunctionDict["eventAndTime"]
        rates = []
        for pop in (trial.treatedPop, trial.controlPop):
            pairs = eventAndTimeFunc(pop)
            events = sum(int(event) for event, _ in pairs)
            totalPY = sum(personYears for _, personYears in pairs)
            #nan when an arm has no person-years, so it cannot be mistaken for an observed zero rate
            rates.append(1000.0 * events / totalPY if totalPY > 0 else float('nan'))
        return tuple(rates)
