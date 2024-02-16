from enum import Enum
from microsim.risk_factor import StaticRiskFactorsType

class PopulationRepositoryType(Enum):
    STATIC_RISK_FACTORS = "staticRiskFactors"
    DYNAMIC_RISK_FACTORS = "dynamicRiskFactors"
    DEFAULT_TREATMENTS = "defaultTreatments"
    OUTCOMES = "outcomes"

class PopulationModelRepository:
    """This class holds the rules for predicting the future of a Population-instance.
       The Population-instance knows these rules but Person-instances do not...Person-objects use these in their
       advance methods where the rules for predicting the future are provided as arguments in the advance methods.
       The rules/models a Population-instance needs are the ones for static risk factors (trivial but for consistency still needed,
       dynamic risk factors, default treatments, outcomes. 
       These are reflected in the PopulationRepositoryType class as well."""

    def __init__(self, dynamicRiskFactorRepository, defaultTreatmentRepository, outcomeRepository, staticRiskFactorRepository):
        #setattr(self, "_"+PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value+"Repository", dynamicRiskFactorRepository)
        #setattr(self, "_"+PopulationRepositoryType.DEFAULT_TREATMENTS.value+"Repository", defaultTreatmentRepository)
        #setattr(self, "_"+PopulationRepositoryType.OUTCOMES.value+"Repository", outcomeRepository)
        #setattr(self, "_"+PopulationRepositoryType.STATIC_RISK_FACTORS.value+"Repository", staticRiskFactorRepository)
        self._repository = {
            PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value: dynamicRiskFactorRepository,
            PopulationRepositoryType.DEFAULT_TREATMENTS.value: defaultTreatmentRepository,
            PopulationRepositoryType.OUTCOMES.value: outcomeRepository,
            PopulationRepositoryType.STATIC_RISK_FACTORS.value: staticRiskFactorRepository}
        
        
