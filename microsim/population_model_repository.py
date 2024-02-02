from enum import Enum
from microsim.risk_factor import StaticRiskFactorsType

class PopulationRepositoryType(Enum):
    STATIC_RISK_FACTORS = "staticRiskFactors"
    DYNAMIC_RISK_FACTORS = "dynamicRiskFactors"
    DEFAULT_TREATMENTS = "defaultTreatments"
    OUTCOMES = "outcomes"

class PopulationModelRepository:

    def __init__(self, dynamicRiskFactorRepository, defaultTreatmentRepository, outcomeRepository, staticRiskFactorRepository):
        setattr(self, "_"+PopulationRepositoryType.DYNAMIC_RISK_FACTORS.value+"Repository", dynamicRiskFactorRepository)
        setattr(self, "_"+PopulationRepositoryType.DEFAULT_TREATMENTS.value+"Repository", defaultTreatmentRepository)
        setattr(self, "_"+PopulationRepositoryType.OUTCOMES.value+"Repository", outcomeRepository)
        setattr(self, "_"+PopulationRepositoryType.STATIC_RISK_FACTORS.value+"Repository", staticRiskFactorRepository)
        
        
