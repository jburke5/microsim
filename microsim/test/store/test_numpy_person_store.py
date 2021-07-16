from dataclasses import dataclass
from itertools import product
from unittest import TestCase
from microsim.store.numpy_person_store import NumpyPersonStore
from microsim.store.dataclass_numpy_data_converter import DataclassNumpyDataConverter


@dataclass
class StaticTestData:
    starting_age: int


@dataclass
class DynamicTestData:
    age: int


@dataclass
class EventTestData:
    died: bool = False


class TestNumpyPersonStore(TestCase):
    def setUp(self):
        self._static_data = [StaticTestData(20), StaticTestData(40), StaticTestData(80)]
        self._static_data_converter = DataclassNumpyDataConverter(StaticTestData)

        self._dynamic_data = [DynamicTestData(s.starting_age) for s in self._static_data]
        self._dynamic_data_converter = DataclassNumpyDataConverter(DynamicTestData)

        self._event_data = [EventTestData() for _ in range(len(self._static_data))]
        self._event_data_converter = DataclassNumpyDataConverter(EventTestData)

    def test_get_num_persons_canonical_returns_num_persons(self):
        store = NumpyPersonStore(
            self._static_data,
            self._static_data_converter,
            self._dynamic_data,
            self._dynamic_data_converter,
            self._event_data,
            self._event_data_converter,
        )
        expected_num_persons = 3

        actual_num_persons = store.get_num_persons()
        self.assertEqual(expected_num_persons, actual_num_persons)

    def test_init_data_length_mismatch_raises_error(self):
        # exhaustively test all permutations with mismatched lengths for 3 list of length 3
        mistmatched_lengths = [
            (i + 1, j + 1, k + 1) for i, j, k in product(range(3), repeat=3) if not (i == j == k)
        ]

        for i, j, k in mistmatched_lengths:
            static_data = self._static_data[:i]
            dynamic_data = self._static_data[:j]
            event_data = self._event_data[:k]
            expected_msg = (
                "Lengths of `static_data`, `dynamic_data`, and `event_data` args do not match:"
                f" {i}, {j}, {k}"
            )

            with self.assertRaises(ValueError, msg=expected_msg):
                NumpyPersonStore(
                    static_data,
                    self._static_data_converter,
                    dynamic_data,
                    self._dynamic_data_converter,
                    event_data,
                    self._event_data_converter,
                )