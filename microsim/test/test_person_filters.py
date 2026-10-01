"""Tests for PersonFilter and PersonFilterFactory: the filter container operations,
the named-filter registry, and the behavior of each pre-defined filter."""

import unittest

from microsim.default_treatments.default_treatments import DefaultTreatmentsType
from microsim.outcomes.cv_model_repository import CVModelRepository
from microsim.outcomes.outcome import Outcome, OutcomeType
from microsim.risk_factors.gender import NHANESGender
from microsim.risk_factors.smoking_status import SmokingStatus
from microsim.person.person_factory import PersonFactory
from microsim.person.person_filter import PersonFilter
from microsim.person.person_filter_factory import PersonFilterFactory
from microsim.risk_factors.risk_factor import DynamicRiskFactorsType
from microsim.test.test_person_factory import build_kaiser_row


class TestPersonFilter(unittest.TestCase):
    def test_init_has_empty_df_and_person_levels(self):
        pf = PersonFilter()
        self.assertEqual(pf.filters, {"df": {}, "person": {}})

    def test_add_filter(self):
        pf = PersonFilter()
        fn = lambda x: x[DynamicRiskFactorsType.AGE.value] >= 18
        pf.add_filter("df", "adult", fn)
        self.assertIs(pf.filters["df"]["adult"], fn)
        self.assertEqual(pf.filters["person"], {})

    def test_add_filter_defaults(self):
        pf = PersonFilter()
        pf.add_filter()
        self.assertTrue(pf.filters["person"]["all"](object()))

    def test_add_filter_overwrites_same_name(self):
        pf = PersonFilter()
        pf.add_filter("df", "adult", lambda x: True)
        replacement = lambda x: False
        pf.add_filter("df", "adult", replacement)
        self.assertIs(pf.filters["df"]["adult"], replacement)

    def test_rm_filter(self):
        pf = PersonFilter()
        pf.add_filter("df", "adult", lambda x: True)
        pf.rm_filter("df", "adult")
        self.assertNotIn("adult", pf.filters["df"])

    def test_str_and_repr_list_filters(self):
        pf = PersonFilter()
        pf.add_filter("df", "adult", lambda x: True)
        self.assertIn("df", str(pf))
        self.assertIn("adult", str(pf))
        self.assertEqual(repr(pf), str(pf))


class TestGetPersonFilter(unittest.TestCase):
    def test_default_is_adult_only(self):
        pf = PersonFilterFactory.get_person_filter()
        self.assertEqual(set(pf.filters["df"].keys()), {"adult"})
        self.assertEqual(pf.filters["person"], {})

    def test_empty_list_returns_no_filters(self):
        pf = PersonFilterFactory.get_person_filter([])
        self.assertEqual(pf.filters, {"df": {}, "person": {}})

    def test_filters_added_at_registered_levels(self):
        pf = PersonFilterFactory.get_person_filter(["lowSBPLimit", "highCVLimit"])
        self.assertEqual(set(pf.filters["df"].keys()), {"lowSBPLimit"})
        self.assertEqual(set(pf.filters["person"].keys()), {"highCVLimit"})

    def test_unknown_filter_raises(self):
        with self.assertRaises(ValueError) as ctx:
            PersonFilterFactory.get_person_filter(["notAFilter"])
        self.assertIn("adult", str(ctx.exception))


class TestDfLevelRegistryFilters(unittest.TestCase):
    def _filter(self, name):
        return PersonFilterFactory.filterMap[name][1]

    def test_adult(self):
        adult = self._filter("adult")
        self.assertTrue(adult({DynamicRiskFactorsType.AGE.value: 18}))
        self.assertFalse(adult({DynamicRiskFactorsType.AGE.value: 17}))

    def test_lowSBPLimit(self):
        lowSBP = self._filter("lowSBPLimit")
        self.assertTrue(lowSBP({DynamicRiskFactorsType.SBP.value: 127}))
        self.assertFalse(lowSBP({DynamicRiskFactorsType.SBP.value: 126}))

    def test_lowDBPLimit(self):
        lowDBP = self._filter("lowDBPLimit")
        self.assertTrue(lowDBP({DynamicRiskFactorsType.DBP.value: 86}))
        self.assertFalse(lowDBP({DynamicRiskFactorsType.DBP.value: 85}))

    def test_highAntiHypertensivesLimit(self):
        highAntiHtn = self._filter("highAntiHypertensivesLimit")
        self.assertTrue(highAntiHtn({DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 3}))
        self.assertFalse(highAntiHtn({DefaultTreatmentsType.ANTI_HYPERTENSIVE_COUNT.value: 4}))


class TestPersonLevelRegistryFilters(unittest.TestCase):
    """Kaiser persons carry the cognition, WMH, and (possibly) epilepsy outcomes
       the person-level filters need."""

    @classmethod
    def setUpClass(cls):
        cls._person = PersonFactory.get_kaiser_person(build_kaiser_row())

    def _filter(self, name):
        return PersonFilterFactory.filterMap[name][1]

    def test_hasEpilepsy(self):
        self.assertEqual(self._filter("hasEpilepsy")(self._person), self._person.has_epilepsy())

    def test_noMCI(self):
        self.assertEqual(self._filter("noMCI")(self._person),
                         not self._person.has_mci(inSim=False))

    def test_highCVLimit(self):
        risk = CVModelRepository().select_outcome_model_for_person(self._person)\
                                  .get_risk_for_person(self._person)
        self.assertEqual(self._filter("highCVLimit")(self._person), risk < 0.00477)


class TestPersonFilterContract(unittest.TestCase):
    """Behavioral guarantees of the filter container, independent of implementation."""

    def test_instances_do_not_share_state(self):
        pf1 = PersonFilter()
        pf1.add_filter("df", "adult", lambda x: True)
        pf2 = PersonFilter()
        self.assertEqual(pf2.filters, {"df": {}, "person": {}})

    def test_levels_are_independent(self):
        pf = PersonFilter()
        pf.add_filter("df", "sameName", lambda x: True)
        pf.add_filter("person", "sameName", lambda x: False)
        self.assertTrue(pf.filters["df"]["sameName"](None))
        self.assertFalse(pf.filters["person"]["sameName"](None))
        pf.rm_filter("df", "sameName")
        self.assertIn("sameName", pf.filters["person"])

    def test_rm_missing_filter_fails_loudly(self):
        with self.assertRaises(KeyError):
            PersonFilter().rm_filter("df", "neverAdded")

    def test_unknown_level_fails_loudly(self):
        with self.assertRaises(KeyError):
            PersonFilter().add_filter("dataframe", "adult", lambda x: True)

    def test_readd_after_remove(self):
        pf = PersonFilter()
        pf.add_filter("df", "adult", lambda x: True)
        pf.rm_filter("df", "adult")
        pf.add_filter("df", "adult", lambda x: False)
        self.assertFalse(pf.filters["df"]["adult"](None))


class TestPersonFilterFactoryContract(unittest.TestCase):
    """The factory must hand out fresh, independent PersonFilters and never let a caller
       mutate the shared registry."""

    def test_returns_fresh_instances(self):
        pf1 = PersonFilterFactory.get_person_filter()
        pf1.rm_filter("df", "adult")
        pf2 = PersonFilterFactory.get_person_filter()
        self.assertIn("adult", pf2.filters["df"])

    def test_customizing_a_filter_does_not_touch_the_registry(self):
        pf = PersonFilterFactory.get_person_filter(["adult"])
        pf.add_filter("df", "custom", lambda x: True)
        self.assertNotIn("custom", PersonFilterFactory.filterMap)
        self.assertEqual(set(PersonFilterFactory.get_person_filter(["adult"]).filters["df"]),
                         {"adult"})

    def test_duplicate_names_collapse(self):
        pf = PersonFilterFactory.get_person_filter(["adult", "adult"])
        self.assertEqual(len(pf.filters["df"]), 1)

    def test_adult_partitions_exactly_at_eighteen(self):
        adult = PersonFilterFactory.filterMap["adult"][1]
        for age in range(0, 101):
            self.assertEqual(adult({DynamicRiskFactorsType.AGE.value: age}), age >= 18)


class TestPersonLevelFilterLogic(unittest.TestCase):
    """The person-level filters must keep/drop people according to their baseline state."""

    def test_highCVLimit_keeps_low_risk_and_drops_high_risk(self):
        highCV = PersonFilterFactory.filterMap["highCVLimit"][1]
        healthy = PersonFactory.get_kaiser_person(build_kaiser_row(
            age=40, gender=NHANESGender.FEMALE.value, smokingStatus=SmokingStatus.NEVER.value,
            sbp=105, dbp=70, a1c=5.0, hdl=70, ldl=80, trig=90, totChol=160, bmi=22,
            creatinine=0.7, anyPhysicalActivity=True, afib=False, pvd=False,
            statin=0, antiHypertensiveCount=0))
        sick = PersonFactory.get_kaiser_person(build_kaiser_row(
            age=85, gender=NHANESGender.MALE.value, smokingStatus=SmokingStatus.CURRENT.value,
            sbp=180, dbp=95, a1c=10., hdl=30, ldl=180, trig=300, totChol=280, bmi=35,
            creatinine=1.4, anyPhysicalActivity=False, afib=True, pvd=True,
            statin=1, antiHypertensiveCount=3))
        self.assertTrue(highCV(healthy), "a healthy 40-year-old should be under the CV risk limit")
        self.assertFalse(highCV(sick), "a sick 85-year-old should exceed the CV risk limit")

    def test_hasEpilepsy_tracks_baseline_epilepsy(self):
        hasEpilepsy = PersonFilterFactory.filterMap["hasEpilepsy"][1]
        person = PersonFactory.get_kaiser_person(build_kaiser_row())
        person._outcomes[OutcomeType.EPILEPSY].clear()
        self.assertFalse(hasEpilepsy(person))
        person.add_outcome(Outcome(OutcomeType.EPILEPSY, fatal=False, priorToSim=True))
        self.assertTrue(hasEpilepsy(person))


if __name__ == "__main__":
    unittest.main()
