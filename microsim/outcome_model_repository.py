from microsim.outcome import OutcomeType
from microsim.dementia_model_repository import DementiaModelRepository
from microsim.gcp_model_repository import GCPModelRepository
from microsim.qaly_model_repository import QALYModelRepository
from microsim.cv_model_repository import CVModelRepository
from microsim.stroke_partition_model_repository import StrokePartitionModelRepository
from microsim.mi_partition_model_repository import MIPartitionModelRepository
from microsim.non_cv_model_repository import NonCVModelRepository
from microsim.death_model_repository import DeathModelRepository

class OutcomeModelRepository:
    """Holds the rules for all outcomes.
       Via a dictionary, this object selects the appropriate model repository for an outcome.
       The model repository will then select the appropriate model for a Person-instance (via a select_outcome_model_for_person function).
       The model then obtains the outcome for the Person-instance (via a get_next_outcome function)."""
    def __init__(self):
        self._modelRepository = {OutcomeType.DEMENTIA: DementiaModelRepository(),
                          OutcomeType.GLOBAL_COGNITIVE_PERFORMANCE: GCPModelRepository(),
                          OutcomeType.QUALITYADJUSTED_LIFE_YEARS: QALYModelRepository(),
                          OutcomeType.CARDIOVASCULAR: CVModelRepository(),
                          OutcomeType.MI: StrokePartitionModelRepository(),
                          OutcomeType.STROKE: MIPartitionModelRepository(),
                          OutcomeType.NONCARDIOVASCULAR: NonCVModelRepository(),
                          OutcomeType.DEATH: DeathModelRepository()}
        #must have a model repository for all outcome types
        self.check_repository_completeness()
 
    #I wonder if this is more appropriate for a test function       
    def check_repository_completeness(self):
        for outcome in OutcomeType:
            if outcome not in list(self._modelRepository.keys()):
                raise RuntimeError("OutcomeModelRepository is incomplete")


