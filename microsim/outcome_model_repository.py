from microsim.outcome import OutcomeType
from microsim.dementia_model_repository import DementiaModelRepository
from microsim.cognition_model_repository import CognitionModelRepository
from microsim.qaly_model_repository import QALYModelRepository
from microsim.cv_model_repository import CVModelRepository
from microsim.stroke_partition_model_repository import StrokePartitionModelRepository
from microsim.mi_partition_model_repository import MIPartitionModelRepository
from microsim.non_cv_model_repository import NonCVModelRepository
from microsim.death_model_repository import DeathModelRepository
from microsim.ci_model_repository import CIModelRepository

class OutcomeModelRepository:
    """Holds the rules for all outcomes.
       Via a dictionary, this object selects the appropriate model repository for an outcome.
       The model repository will then select the appropriate model for a Person-instance (via a select_outcome_model_for_person function).
       The model then obtains the outcome for the Person-instance (via a get_next_outcome function).
       Outcomes are Outcome-instances when the only information we want is the occurence of the outcome, age, and fatality.
       Examples are death outcomes, mi outcomes.
       Outcomes are Outcome subclasses, eg StrokeOutcome, when more information about the outcome need to be stored, an outcome phenotype.
       Examples are StrokeOutcome (nihss, type etc), GCPOutcome (gcp), QALYOutcome (qaly)."""
    def __init__(self):
        self._repository = {OutcomeType.DEMENTIA: DementiaModelRepository(),
                          OutcomeType.COGNITION: CognitionModelRepository(),
                          OutcomeType.CI: CIModelRepository(),
                          OutcomeType.QUALITYADJUSTED_LIFE_YEARS: QALYModelRepository(),
                          OutcomeType.CARDIOVASCULAR: CVModelRepository(),
                          OutcomeType.MI: MIPartitionModelRepository(),
                          OutcomeType.STROKE: StrokePartitionModelRepository(),
                          OutcomeType.NONCARDIOVASCULAR: NonCVModelRepository(),
                          OutcomeType.DEATH: DeathModelRepository()}
        #must have a model repository for all outcome types
        self.check_repository_completeness()
 
    def check_repository_completeness(self):
        for outcome in OutcomeType:
            if outcome not in list(self._repository.keys()):
                raise RuntimeError("OutcomeModelRepository is incomplete")


