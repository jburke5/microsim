from dataclasses import dataclass
import math
from unittest import TestCase
from microsim.store.dataclass_numpy_data_converter import DataclassNumpyDataConverter
from microsim.test.helper.partial_alphabet import PartialAlphabet
from microsim.test.helper.other_type import OtherType


@dataclass
class CanonicalTestData:
    is_true: bool
    my_int: int
    my_float: float
    my_letter: PartialAlphabet


class TestDataclassNumpyConverter(TestCase):
    def test_init_type_not_dataclass_raises_type_error(self):
        with self.assertRaises(TypeError, msg="Given argument `type` is not a dataclass: int"):
            DataclassNumpyDataConverter(int)

    def test_get_dtype_canonical(self):
        converter = DataclassNumpyDataConverter(CanonicalTestData)
        expected_names = ["is_true", "my_int", "my_float", "my_letter"]

        dtype = converter.get_dtype()

        actual_names = list(dtype.names)
        self.assertListEqual(expected_names, actual_names)

    def test_to_row_tuple_different_type_raises_type_error(self):
        converter = DataclassNumpyDataConverter(CanonicalTestData)
        obj = OtherType()
        expected_msg = (
            "Given argument `obj` is not an instance of the configured type (CanonicalTestData):"
            " OtherType"
        )

        with self.assertRaises(TypeError, msg=expected_msg):
            converter.to_row_tuple(obj)

    def test_to_row_tuple_canonical(self):
        converter = DataclassNumpyDataConverter(CanonicalTestData)
        obj = CanonicalTestData(True, 1, math.pi, PartialAlphabet.A)
        expected = (True, 1, math.pi, PartialAlphabet.A)

        actual = converter.to_row_tuple(obj)

        self.assertEqual(expected, actual)
