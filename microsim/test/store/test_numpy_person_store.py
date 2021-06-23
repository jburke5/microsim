from dataclasses import dataclass
from unittest import TestCase
from microsim.store.numpy_person_store import NumpyPersonStore
from microsim.store.dataclass_numpy_data_converter import DataclassNumpyDataConverter


@dataclass
class StaticTestData:
    starting_age: int


class TestNumpyPersonStore(TestCase):
    def test_get_num_persons_canonical_returns_num_persons(self):
        data = [StaticTestData(20), StaticTestData(40), StaticTestData(80)]
        static_data_converter = DataclassNumpyDataConverter(StaticTestData)
        store = NumpyPersonStore(data, static_data_converter)
        expected_num_persons = 3

        actual_num_persons = store.get_num_persons()

        self.assertEqual(expected_num_persons, actual_num_persons)
