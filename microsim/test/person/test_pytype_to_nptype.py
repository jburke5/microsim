from enum import IntEnum
import unittest
import numpy as np
from microsim.person.pytype_to_nptype import pytype_to_nptype


class PartialAlphabet(IntEnum):
    A = 1
    B = 2
    C = 3


class OtherType:
    pass


class TestPytypeToNptype(unittest.TestCase):
    def test_bool_to_bool_(self):
        pytype = bool

        nptype = pytype_to_nptype(pytype)

        self.assertIs(np.bool_, nptype)

    def test_int_to_int64(self):
        pytype = int

        nptype = pytype_to_nptype(pytype)

        self.assertIs(np.int64, nptype)

    def test_float_to_float64(self):
        pytype = float

        nptype = pytype_to_nptype(pytype)

        self.assertIs(np.float64, nptype)

    def test_int_enum_to_int64(self):
        pytype = IntEnum

        nptype = pytype_to_nptype(pytype)

        self.assertIs(np.int64, nptype)

    def test_int_enum_subclass_to_int64(self):
        pytype = PartialAlphabet

        nptype = pytype_to_nptype(pytype)

        self.assertIs(np.int64, nptype)

    def test_other_type_raises_not_implemented_error(self):
        pytype = OtherType

        def to_nptype():
            pytype_to_nptype(pytype)

        self.assertRaises(NotImplementedError, to_nptype)
