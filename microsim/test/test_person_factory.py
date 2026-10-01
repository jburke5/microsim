"""Tests for PersonFactory: dispatch, init-information organization, and full construction
from synthetic NHANES and Kaiser rows. Person class functions are covered elsewhere."""

import unittest
import pandas as pd

from microsim.common.population_type import PopulationType
from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.outcomes.outcome import OutcomeType
from microsim.outcomes.outcome_prevalence_model_repository import OutcomePrevalenceModelRepository
from microsim.person.person import Person
from microsim.person.person_factory import PersonFactory
from microsim.risk_factors.alcohol_category import AlcoholCategory
from microsim.risk_factors.education import Education
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.initialization_model_repository import InitializationModelRepository
from microsim.risk_factors.modality import Modality
from microsim.risk_factors.race_ethnicity import RaceEthnicity
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType, StaticRiskFactorsType
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.treatment_strategies.treatment_strategies import TreatmentStrategiesType


def build_nhanes_row(**overrides):
    row = {
        "name": "testNhanesPerson",
        DynamicRiskFactorsType.AGE.value: 60,
        StaticRiskFactorsType.GENDER.value: NHANESGender.MALE.value,
        StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
        StaticRiskFactorsType.EDUCATION.value: Education.COLLEGEGRADUATE.value,
        StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
        DynamicRiskFactorsType.SBP.value: 120,
        DynamicRiskFactorsType.DBP.value: 80,
        DynamicRiskFactorsType.A1C.value: 5.5,
        DynamicRiskFactorsType.HDL.value: 50,
        DynamicRiskFactorsType.LDL.value: 90,
        DynamicRiskFactorsType.TRIG.value: 150,
        DynamicRiskFactorsType.TOT_CHOL.value: 200,
        DynamicRiskFactorsType.BMI.value: 25,
        DynamicRiskFactorsType.WAIST.value: 90,
        DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: False,
        DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value: 0.,  #drinks/week
        DynamicRiskFactorsType.CREATININE.value: 0.9,
        DefaultTreatmentsType.STATIN.value: 0,
        DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 0,
    }
    row.update(overrides)
    return pd.Series(row)


def build_kaiser_row(**overrides):
    row = {
        "name": "testKaiserPerson",
        StaticRiskFactorsType.MODALITY.value: Modality.CT.value,
        StaticRiskFactorsType.GENDER.value: NHANESGender.FEMALE.value,
        StaticRiskFactorsType.RACE_ETHNICITY.value: RaceEthnicity.NON_HISPANIC_WHITE.value,
        StaticRiskFactorsType.SMOKING_STATUS.value: SmokingStatus.NEVER.value,
        DynamicRiskFactorsType.AGE.value: 70,
        DynamicRiskFactorsType.SBP.value: 130,
        DynamicRiskFactorsType.DBP.value: 75,
        DynamicRiskFactorsType.A1C.value: 5.8,
        DynamicRiskFactorsType.HDL.value: 55,
        DynamicRiskFactorsType.LDL.value: 100,
        DynamicRiskFactorsType.TRIG.value: 140,
        DynamicRiskFactorsType.TOT_CHOL.value: 210,
        DynamicRiskFactorsType.BMI.value: 27,
        DynamicRiskFactorsType.CREATININE.value: 0.8,
        DynamicRiskFactorsType.ANY_PHYSICAL_ACTIVITY.value: True,
        DynamicRiskFactorsType.AFIB.value: False,
        DynamicRiskFactorsType.PVD.value: False,
        DefaultTreatmentsType.STATIN.value: 1,
        DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 1,
    }
    row.update(overrides)
    return pd.Series(row)


class TestGetPersonDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._imr = InitializationModelRepository()

    def test_dispatches_to_nhanes(self):
        person = PersonFactory.get_person(
            build_nhanes_row(), popType=PopulationType.NHANES.value,
            initializationModelRepository=self._imr)
        self.assertIsInstance(person, Person)
        self.assertEqual(person._modality, Modality.NO.value)

    def test_dispatches_to_kaiser(self):
        person = PersonFactory.get_person(
            build_kaiser_row(), popType=PopulationType.KAISER.value,
            initializationModelRepository=self._imr)
        self.assertIsInstance(person, Person)
        self.assertEqual(person._modality, Modality.CT.value)

    def test_unknown_population_type_raises(self):
        with self.assertRaises(RuntimeError):
            PersonFactory.get_person(build_nhanes_row(), popType="notAPopulation")


class TestGetNhanesPersonInitInformation(unittest.TestCase):
    def test_organizes_row_into_init_dicts(self):
        (name, static, dynamic, treatments, strategies, outcomes) = \
            PersonFactory.get_nhanes_person_init_information(build_nhanes_row())
        self.assertEqual(name, "testNhanesPerson")
        self.assertIsInstance(static[StaticRiskFactorsType.RACE_ETHNICITY.value], RaceEthnicity)
        self.assertIsInstance(static[StaticRiskFactorsType.EDUCATION.value], Education)
        self.assertIsInstance(static[StaticRiskFactorsType.GENDER.value], NHANESGender)
        self.assertIsInstance(static[StaticRiskFactorsType.SMOKING_STATUS.value], SmokingStatus)
        self.assertIsNone(static[StaticRiskFactorsType.MODALITY.value])
        self.assertEqual(dynamic[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value], 0.)  #drinks/week
        self.assertIsNone(dynamic[DynamicRiskFactorsType.AFIB.value])
        self.assertIsNone(dynamic[DynamicRiskFactorsType.PVD.value])
        self.assertEqual(dynamic[DynamicRiskFactorsType.SBP.value], 120)
        self.assertEqual(treatments[DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value], 0)
        self.assertEqual(set(strategies.keys()), {ts.value for ts in TreatmentStrategiesType})
        for strategy in strategies.values():
            self.assertEqual(strategy, {"status": None})
        self.assertEqual(set(outcomes.keys()), set(OutcomeType))
        for outcomeList in outcomes.values():
            self.assertEqual(outcomeList, [])

    def test_statin_is_converted_to_bool(self):
        #NHANES includes statin=2; the models expect a 0/1 indicator
        for rawStatin, expected in [(0, False), (1, True), (2, True)]:
            (_, _, _, treatments, _, _) = PersonFactory.get_nhanes_person_init_information(
                build_nhanes_row(statin=rawStatin))
            self.assertIs(treatments[DefaultTreatmentsType.STATIN.value], expected)

    def test_adult_bounds_applied(self):
        (_, _, dynamic, _, _, _) = PersonFactory.get_nhanes_person_init_information(
            build_nhanes_row(**{DynamicRiskFactorsType.SBP.value: 500}))
        self.assertEqual(dynamic[DynamicRiskFactorsType.SBP.value], 297.)

    def test_child_bounds_applied(self):
        (_, _, dynamic, _, _, _) = PersonFactory.get_nhanes_person_init_information(
            build_nhanes_row(**{DynamicRiskFactorsType.AGE.value: 10,
                                DynamicRiskFactorsType.SBP.value: 500}))
        self.assertEqual(dynamic[DynamicRiskFactorsType.SBP.value], 190.3)


class TestGetNhanesPerson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._imr = InitializationModelRepository()

    def test_initializes_model_based_attributes(self):
        person = PersonFactory.get_nhanes_person(build_nhanes_row(), self._imr)
        self.assertEqual(person._name, "testNhanesPerson")
        self.assertEqual(len(person._pvd), 1)
        self.assertIn(person._pvd[0], (True, False))
        self.assertEqual(len(person._afib), 1)
        self.assertIn(person._afib[0], (True, False))
        self.assertEqual(person._modality, Modality.NO.value)

    def test_no_prevalence_seeding_when_repository_omitted(self):
        person = PersonFactory.get_nhanes_person(build_nhanes_row(), self._imr)
        for outcomeList in person._outcomes.values():
            self.assertEqual(outcomeList, [])

    def test_prevalence_seeding_with_repository(self):
        person = PersonFactory.get_nhanes_person(
            build_nhanes_row(), self._imr,
            outcomePrevalenceModelRepository=OutcomePrevalenceModelRepository())
        #cognition is a continuous score, so its prevalence model always seeds an outcome
        cognition = person._outcomes[OutcomeType.COGNITION]
        self.assertEqual(len(cognition), 1)
        age, outcome = cognition[0]
        self.assertIsNone(age)
        self.assertTrue(outcome.priorToSim)


class TestGetKaiserPersonInitInformation(unittest.TestCase):
    def test_organizes_row_into_init_dicts(self):
        (name, static, dynamic, treatments, strategies, outcomes) = \
            PersonFactory.get_kaiser_person_init_information(build_kaiser_row())
        self.assertEqual(name, "testKaiserPerson")
        self.assertEqual(static[StaticRiskFactorsType.MODALITY.value], Modality.CT.value)
        self.assertIsNone(static[StaticRiskFactorsType.EDUCATION.value])
        self.assertIsInstance(static[StaticRiskFactorsType.GENDER.value], NHANESGender)
        self.assertIsNone(dynamic[DynamicRiskFactorsType.WAIST.value])
        self.assertIsNone(dynamic[DynamicRiskFactorsType.ALCOHOL_PER_WEEK.value])
        self.assertEqual(dynamic[DynamicRiskFactorsType.SBP.value], 130)
        self.assertIs(treatments[DefaultTreatmentsType.STATIN.value], True)
        self.assertEqual(set(strategies.keys()), {ts.value for ts in TreatmentStrategiesType})
        self.assertEqual(set(outcomes.keys()), set(OutcomeType))
        for outcomeList in outcomes.values():
            self.assertEqual(outcomeList, [])


class TestGetKaiserPerson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._person = PersonFactory.get_kaiser_person(build_kaiser_row())

    def test_fills_missing_risk_factors(self):
        self.assertEqual(len(self._person._waist), 1)
        self.assertIsNotNone(self._person._waist[0])
        self.assertEqual(len(self._person._alcoholPerWeek), 1)
        self.assertIsNotNone(self._person._alcoholPerWeek[0])
        self.assertIsInstance(self._person._education, Education)

    def test_seeds_outcomes(self):
        self.assertEqual(len(self._person._outcomes[OutcomeType.WMH]), 1)
        self.assertEqual(len(self._person._outcomes[OutcomeType.COGNITION]), 1)
        #epilepsy seeding is probabilistic, so presence is 0 or 1
        self.assertIn(len(self._person._outcomes[OutcomeType.EPILEPSY]), (0, 1))

    def test_accepts_shared_initialization_model_repository(self):
        person = PersonFactory.get_kaiser_person(build_kaiser_row(), InitializationModelRepository())
        self.assertIsInstance(person._education, Education)
        self.assertIsNotNone(person._waist[0])


class TestPersonConstructionContract(unittest.TestCase):
    """Behavioral invariants any fully constructed person must satisfy, regardless of
       how the factory is implemented."""

    @classmethod
    def setUpClass(cls):
        cls._imr = InitializationModelRepository()

    def test_nhanes_person_starts_at_baseline_wave(self):
        person = PersonFactory.get_nhanes_person(build_nhanes_row(), self._imr)
        self.assertEqual(person._waveCompleted, -1)
        self.assertTrue(person.is_alive)
        for rf in person._dynamicRiskFactors:
            values = getattr(person, "_" + rf)
            self.assertEqual(len(values), 1, f"{rf} should hold only the baseline value")
            self.assertIsNotNone(values[0], f"{rf} baseline should be initialized")
        for treatment in person._defaultTreatments:
            self.assertEqual(len(getattr(person, "_" + treatment)), 1)

    def test_kaiser_person_starts_at_baseline_wave(self):
        person = PersonFactory.get_kaiser_person(build_kaiser_row())
        self.assertEqual(person._waveCompleted, -1)
        self.assertTrue(person.is_alive)
        for rf in person._dynamicRiskFactors:
            values = getattr(person, "_" + rf)
            self.assertEqual(len(values), 1, f"{rf} should hold only the baseline value")
            self.assertIsNotNone(values[0], f"{rf} baseline should be initialized")

    def test_static_risk_factors_fully_initialized(self):
        nhanesPerson = PersonFactory.get_nhanes_person(build_nhanes_row(), self._imr)
        kaiserPerson = PersonFactory.get_kaiser_person(build_kaiser_row())
        for person in (nhanesPerson, kaiserPerson):
            for rf in person._staticRiskFactors:
                self.assertIsNotNone(getattr(person, "_" + rf),
                                     f"static risk factor {rf} should be initialized")

    def test_seeded_outcomes_are_prior_to_sim_only(self):
        person = PersonFactory.get_nhanes_person(
            build_nhanes_row(), self._imr,
            outcomePrevalenceModelRepository=OutcomePrevalenceModelRepository())
        for outcomeType in OutcomeType:
            self.assertFalse(person.has_outcome(outcomeType, inSim=True),
                             f"{outcomeType} should not count as an in-sim event")
            for age, outcome in person._outcomes[outcomeType]:
                self.assertIsNone(age)
                self.assertTrue(outcome.priorToSim)

    def test_persons_from_same_row_are_independent(self):
        row = build_nhanes_row()
        person1 = PersonFactory.get_nhanes_person(row, self._imr)
        person2 = PersonFactory.get_nhanes_person(row, self._imr)
        self.assertIsNot(person1._rng, person2._rng)
        self.assertIsNot(person1._age, person2._age)
        person1._age.append(61)
        self.assertEqual(len(person2._age), 1)

    def test_name_not_shadowed_by_series_index(self):
        #a row taken out of a dataframe carries its index label as Series.name
        x = pd.DataFrame([build_nhanes_row()]).iloc[0]
        person = PersonFactory.get_nhanes_person(x, self._imr)
        self.assertEqual(person._name, "testNhanesPerson")

    def test_bounds_clip_in_both_directions(self):
        (_, _, low, _, _, _) = PersonFactory.get_nhanes_person_init_information(
            build_nhanes_row(sbp=1, bmi=1, hdl=1))
        self.assertEqual(low[DynamicRiskFactorsType.SBP.value], 58.20)
        self.assertEqual(low[DynamicRiskFactorsType.BMI.value], 10.836)
        self.assertEqual(low[DynamicRiskFactorsType.HDL.value], 5.4)
        (_, _, high, _, _, _) = PersonFactory.get_nhanes_person_init_information(
            build_nhanes_row(bmi=1000, hdl=10000))
        self.assertEqual(high[DynamicRiskFactorsType.BMI.value], 143.23)
        self.assertEqual(high[DynamicRiskFactorsType.HDL.value], 248.6)

    def test_in_range_values_pass_through_unchanged(self):
        (_, _, dynamic, _, _, _) = PersonFactory.get_nhanes_person_init_information(
            build_nhanes_row())
        self.assertEqual(dynamic[DynamicRiskFactorsType.SBP.value], 120)
        self.assertEqual(dynamic[DynamicRiskFactorsType.BMI.value], 25)
        self.assertEqual(dynamic[DynamicRiskFactorsType.A1C.value], 5.5)

    def test_dispatch_matches_direct_construction(self):
        row = build_nhanes_row()
        viaDispatch = PersonFactory.get_person(row, popType=PopulationType.NHANES.value,
                                               initializationModelRepository=self._imr)
        direct = PersonFactory.get_nhanes_person(row, self._imr)
        self.assertEqual(viaDispatch._name, direct._name)
        self.assertEqual(viaDispatch._age, direct._age)
        self.assertEqual(viaDispatch._gender, direct._gender)

    def test_kaiser_modality_preserved_from_row(self):
        for modality in (Modality.CT.value, Modality.MR.value):
            person = PersonFactory.get_kaiser_person(build_kaiser_row(modality=modality))
            self.assertEqual(person._modality, modality)

    def test_shared_repository_reusable_across_persons_and_populations(self):
        imr = InitializationModelRepository()
        for _ in range(3):
            nhanesPerson = PersonFactory.get_nhanes_person(build_nhanes_row(), imr)
            kaiserPerson = PersonFactory.get_kaiser_person(build_kaiser_row(), imr)
            self.assertIsNotNone(nhanesPerson._modality)
            self.assertIsNotNone(kaiserPerson._education)


if __name__ == "__main__":
    unittest.main()
