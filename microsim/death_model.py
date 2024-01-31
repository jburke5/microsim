from microsim.outcome import Outcome, OutcomeType

class DeathModel:
    """Checks if any outcome produced in the current year was fatal."""

    def __init__(self):
        pass
    
    def generate_next_outcome(self, person):
        return Outcome(OutcomeType.DEATH, True)
    
    def get_next_outcome(self, person):
        deathOutcome = None
        for outcome in list(person._outcomes.keys()):
            if person.has_fatal_outcome_at_current_age(outcome):
                deathOutcome = self.generate_next_outcome(person)
                break
        return deathOutcome
