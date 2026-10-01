from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.cv_model_repository import CVPrevalenceModelRepository
from microsim.outcomes.stroke_partition_model_repository import StrokePrevalenceModelRepository
from microsim.outcomes.mi_partition_model_repository import MIPrevalenceModelRepository
from microsim.outcomes.dementia_model_repository import DementiaPrevalenceModelRepository
from microsim.outcomes.diabetes_model_repository import DiabetesPrevalenceModelRepository
from microsim.outcomes.chronic_kidney_disease_model_repository import (
    ChronicKidneyDiseasePrevalenceModelRepository,
)
from microsim.outcomes.epilepsy_model_repository import EpilepsyPrevalenceModelRepository
from microsim.outcomes.cognition_model_repository import CognitionPrevalenceModelRepository


# Calibrated default priorToSim risk scaling per outcome. Each value is interpreted by the
# corresponding prevalence model: expit-based logistic models treat it as an odds-ratio shift
# (lp + log s); the epilepsy rate model treats it as a direct multiplier. Outcomes whose
# prevalence model ignores riskScaling (MI partition, COGNITION) must not appear here.
#
# To bake in a calibrated value: run PopulationFactory.calibrate_prevalence, then add an
# entry here. For each entry record in a trailing comment: target prevalence, scale and
# target OutcomeTypes, AgeScope, NHANES year and people args used, and date — these define
# the configuration under which the value is exact; off-configuration use is approximate.
#
# Example (do not uncomment unless the value is real):
#   OutcomeType.CARDIOVASCULAR: 1.23,  # scale=CV, target_outcome=CV, target=0.18,
#                                      # AgeScope(65, None), NHANES 1999, 2026-05-15
DEFAULT_PREVALENCE_RISK_SCALING: dict[OutcomeType, float] = {
      # CV and STROKE are a pair from calibration.fit_prevalence_scaling_to_stroke_mi: joint
      # least squares to the GBD stroke and MI rates in Reference.prevalence (ages 50-74, USA,
      # 1999), 2026-08-28. Under them seeded CV sits at the GBD stroke+MI level (the MI
      # partition forces stroke + MI = CV), not the GBD all-CVD level the coefficients target.
      OutcomeType.CARDIOVASCULAR: 0.3474,
      OutcomeType.STROKE: 2.4466,
      # calibrate_prevalence: scale=epilepsy target_outcome=epilepsy scope=pooled_65_plus target=0.0110 scaling=1.8845, CMS-based data for >=65
      OutcomeType.EPILEPSY: 1.88
}


class OutcomePrevalenceModelRepository:
    """Holds the priorToSim (prevalence) rules for outcomes.
       Mirror image of OutcomeModelRepository: every OutcomeType is registered as a key, but
       outcomes without a prevalence model resolve to None and are skipped during seeding.

       riskScaling: optional dict[OutcomeType, float] applied per-outcome inside each
       prevalence model. For expit-based logistic models the scalar is applied as an odds
       shift (lp + log(scaling)); for the epilepsy rate model it is a direct rate multiplier.
       Outcomes without a probabilistic prevalence model (MI partition, cognition GCP) ignore
       riskScaling regardless of what is passed.

       useDefaults: when True (default), entries from DEFAULT_PREVALENCE_RISK_SCALING are
       merged with `riskScaling`; per-outcome values in `riskScaling` override defaults.
       Pass useDefaults=False to bypass the module-level defaults entirely — used by the
       calibrators to measure against a pristine baseline, and by sensitivity analyses
       that want to recover pre-calibration behavior."""

    def __init__(self, riskScaling=None, useDefaults=True):
        if riskScaling is None:
            riskScaling = {}
        if useDefaults:
            effective = {**DEFAULT_PREVALENCE_RISK_SCALING, **riskScaling}
        else:
            effective = dict(riskScaling)
        cvScaling = effective.get(OutcomeType.CARDIOVASCULAR, 1.0)
        strokeScaling = effective.get(OutcomeType.STROKE, 1.0)
        dementiaScaling = effective.get(OutcomeType.DEMENTIA, 1.0)
        epilepsyScaling = effective.get(OutcomeType.EPILEPSY, 1.0)
        diabetesScaling = effective.get(OutcomeType.DIABETES, 1.0)
        ckdScaling = effective.get(OutcomeType.CHRONIC_KIDNEY_DISEASE, 1.0)
        self._repository = {
            OutcomeType.WMH:                      None,
            OutcomeType.COGNITION:                CognitionPrevalenceModelRepository(),
            OutcomeType.CI:                       None,
            OutcomeType.MCI:                      None,
            OutcomeType.DIABETES:                 DiabetesPrevalenceModelRepository(riskScaling=diabetesScaling),
            OutcomeType.CHRONIC_KIDNEY_DISEASE:   ChronicKidneyDiseasePrevalenceModelRepository(riskScaling=ckdScaling),
            OutcomeType.CARDIOVASCULAR:           CVPrevalenceModelRepository(riskScaling=cvScaling),
            OutcomeType.STROKE:                   StrokePrevalenceModelRepository(riskScaling=strokeScaling),
            OutcomeType.MI:                       MIPrevalenceModelRepository(),
            OutcomeType.NONCARDIOVASCULAR:        None,
            OutcomeType.DEMENTIA:                 DementiaPrevalenceModelRepository(riskScaling=dementiaScaling),
            OutcomeType.EPILEPSY:                 EpilepsyPrevalenceModelRepository(riskScaling=epilepsyScaling),
            OutcomeType.DEATH:                    None,
            OutcomeType.QUALITYADJUSTED_LIFE_YEARS: None,
        }
        self.check_repository_completeness()

    def check_repository_completeness(self):
        for outcome in OutcomeType:
            if outcome not in list(self._repository.keys()):
                raise RuntimeError("OutcomePrevalenceModelRepository is incomplete")

    def has_prevalence_model(self, outcomeType):
        return self._repository.get(outcomeType) is not None
