import math

from scipy.special import expit

from microsim.outcomes.outcome import Outcome


class OutcomePrevalenceBase:
    """Base class for prevalence models that seed priorToSim outcomes when a Person is constructed.

    Subclasses MUST implement:
      - get_linear_predictor_for_person(person)
      - calc_linear_predictor_for_patient_characteristics(...)  (signature is subclass-specific)

    Subclasses set the class attribute `_outcomeType`. Subclasses whose Outcome carries
    phenotype data (e.g., StrokeOutcome) override `generate_prevalent_outcome`.

    `_riskScaling` shifts the linear predictor by log(scaling), so the predicted probability
    is `expit(lp + log(scaling))`. This is the odds-ratio interpretation: scaling=2 doubles
    the odds; scaling=1 (default) is a no-op. Subclasses with non-expit risk (e.g., the
    epilepsy rate-per-1000 model) override `get_risk_for_person` to apply scaling directly.
    """

    _outcomeType = None  # subclasses must set this
    _riskScaling = 1.0  # subclasses with scaling support override per-instance via __init__

    def get_risk_for_person(self, person):
        return expit(self.get_linear_predictor_for_person(person) + math.log(self._riskScaling))

    def get_linear_predictor_for_person(self, person):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_linear_predictor_for_person"
        )

    def calc_linear_predictor_for_patient_characteristics(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "calc_linear_predictor_for_patient_characteristics"
        )

    def generate_prevalent_outcome(self, person):
        return Outcome(self._outcomeType, fatal=False, priorToSim=True)

    def get_prevalent_outcome(self, person):
        if person._rng.uniform(size=1) < self.get_risk_for_person(person):
            return self.generate_prevalent_outcome(person)
        return None
