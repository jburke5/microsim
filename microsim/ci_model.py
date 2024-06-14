from microsim.outcome import Outcome, OutcomeType

class CIModel:
    """Cognitive impairment model."""

    def __init__(self):
        pass

    def generate_next_outcome(self, person):
        return Outcome(OutcomeType.CI, False)

    def get_next_outcome(self, person):
        return self.generate_next_outcome(person) if person.get_outcome_item_overall_change(OutcomeType.COGNITION, "gcp") < (-0.5*10.3099) else None   
