from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_factors.afib_model import AFibPrevalenceModel
from microsim.risk_factors.pvd_model import PVDPrevalenceModel
from microsim.risk_factors.waist_model import WaistPrevalenceModel
from microsim.risk_factors.education_model import EducationPrevalenceModel
from microsim.risk_factors.alcohol_model import AlcoholPrevalenceModel
from microsim.risk_factors.modality_model import ModalityPrevalenceModel


class InitializationModelRepository:
    """Holds prevalence models for risk factors that are needed in Microsim simulations
    but are not provided in the data sources used to construct Person objects.

    Used by PersonFactory.get_nhanes_person and get_kaiser_person at construction time
    to seed PVD/AFIB/MODALITY (NHANES) or WAIST/ALCOHOL/EDUCATION (Kaiser)."""

    def __init__(self):
        self._repository = {
            DynamicRiskFactorsType.AFIB.value: AFibPrevalenceModel(),
            DynamicRiskFactorsType.PVD.value: PVDPrevalenceModel(),
            DynamicRiskFactorsType.WAIST.value: WaistPrevalenceModel(),
            StaticRiskFactorsType.EDUCATION.value: EducationPrevalenceModel(),
            DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: AlcoholPrevalenceModel(),
            StaticRiskFactorsType.MODALITY.value: ModalityPrevalenceModel(),
        }

    def __getitem__(self, key):
        return self._repository[key]
