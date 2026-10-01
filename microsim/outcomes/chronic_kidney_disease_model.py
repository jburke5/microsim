from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.outcomes.outcome_prevalence_base import OutcomePrevalenceBase
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.education import Education
from microsim.risk_factors.smoking_status import SmokingStatus


class ChronicKidneyDiseaseModel:
    """Chronic kidney disease outcome model. First detection is gated on GFR < 60; once a CKD outcome
    has been recorded, a new outcome is emitted every wave thereafter regardless of current GFR."""

    def __init__(self):
        pass

    def generate_next_outcome(self, person):
        return Outcome(OutcomeType.CHRONIC_KIDNEY_DISEASE, False)

    def get_next_outcome(self, person):
        if (
            person.has_outcome(OutcomeType.CHRONIC_KIDNEY_DISEASE, inSim=False)
            or person._current_ckd
        ):
            return self.generate_next_outcome(person)
        return None


class ChronicKidneyDiseasePrevalenceModel(OutcomePrevalenceBase):
    """Logistic prevalence model that seeds priorToSim CKD at Person construction.
    Coefficients below are placeholder zeros — replace with fitted odds ratios."""

    _outcomeType = OutcomeType.CHRONIC_KIDNEY_DISEASE

    def __init__(self, riskScaling=1.0):
        self._intercept = 0.0
        self._riskScaling = riskScaling

    def get_risk_for_person(self, person):
        return 0.0

    def get_linear_predictor_for_person(self, person):
        return self.calc_linear_predictor_for_patient_characteristics(
            person._age[-1],
            person._gender,
            person._raceEthnicity,
            person._education,
            person._smokingStatus,
            person._anyPhysicalActivity[-1],
            person._sbp[-1],
            person._dbp[-1],
            person._totChol[-1],
        )

    def calc_linear_predictor_for_patient_characteristics(
        self,
        age,
        gender,
        raceEthnicity,
        education,
        smokingStatus,
        anyPhysicalActivity,
        sbp,
        dbp,
        totChol,
    ):
        xb = self._intercept

        if age < 65:
            xb += 0.0
        elif 65 <= age < 70:
            xb += 0.0  # reference
        elif 70 <= age < 75:
            xb += 0.0
        elif 75 <= age < 80:
            xb += 0.0
        elif age >= 80:
            xb += 0.0

        if gender == NHANESGender.FEMALE:
            xb += 0.0
        elif gender == NHANESGender.MALE:
            xb += 0.0  # reference

        if raceEthnicity == RaceEthnicity.NON_HISPANIC_WHITE:
            xb += 0.0  # reference
        elif raceEthnicity == RaceEthnicity.ASIAN:
            xb += 0.0
        elif raceEthnicity == RaceEthnicity.NON_HISPANIC_BLACK:
            xb += 0.0
        elif (raceEthnicity == RaceEthnicity.MEXICAN_AMERICAN) | (
            raceEthnicity == RaceEthnicity.OTHER_HISPANIC
        ):
            xb += 0.0
        elif raceEthnicity == RaceEthnicity.OTHER:
            xb += 0.0

        if (education == Education.LESSTHANHIGHSCHOOL) | (education == Education.SOMEHIGHSCHOOL):
            xb += 0.0  # reference
        elif education == Education.HIGHSCHOOLGRADUATE:
            xb += 0.0
        elif education == Education.SOMECOLLEGE:
            xb += 0.0
        elif education == Education.COLLEGEGRADUATE:
            xb += 0.0

        if smokingStatus == SmokingStatus.NEVER:
            xb += 0.0  # reference
        elif smokingStatus == SmokingStatus.FORMER:
            xb += 0.0
        elif smokingStatus == SmokingStatus.CURRENT:
            xb += 0.0

        xb += anyPhysicalActivity * 0.0
        xb += sbp * 0.0
        xb += dbp * 0.0
        xb += totChol * 0.0

        return xb
