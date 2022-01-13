import itertools
from tempfile import NamedTemporaryFile, TemporaryFile
from unittest import TestCase
import numpy as np
from microsim.person.bpcog_person_records import (
    BPCOGPersonStaticRecordProtocol,
    BPCOGPersonDynamicRecordProtocol,
)
from microsim.store.numpy_record_mapping import NumpyRecordMapping, NumpyEventRecordMapping
from microsim.store.numpy_person_store import NumpyPersonStore
from microsim.test._validation.helper import BPCOGCohortPersonRecordLoader


class TestNumpyPersonStoreInit(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._num_persons = 1000
        cls._loader_seed = 3308830098  # secrets.randbits(32)  # add `import secrets` to use
        cls._loader = BPCOGCohortPersonRecordLoader(cls._num_persons, 1999, seed=cls._loader_seed)
        assert cls._num_persons == len(cls._loader)
        cls._person_records = list(cls._loader)

    def setUp(self):
        self._static_mapping = NumpyRecordMapping(BPCOGPersonStaticRecordProtocol)
        self._dynamic_mapping = NumpyRecordMapping(BPCOGPersonDynamicRecordProtocol)
        self._event_mapping = NumpyEventRecordMapping()

    def test_npz_file_name_loads_identical_data(self):
        num_ticks = 1
        original_store = NumpyPersonStore(
            self._num_persons,
            num_ticks,
            self._static_mapping,
            self._dynamic_mapping,
            self._event_mapping,
            initial_person_records=self._person_records,
        )
        # need to specify tempfile suffix because Numpy will append `.npz` to filenames that
        # don't have it, writing the array data to a different tempfile, which both causes the test
        # to fail and requires manual cleanup
        with NamedTemporaryFile(suffix=".npz") as f:
            original_store.save_to_file(f.name)
            f.flush()
            f.seek(0)

            loaded_store = NumpyPersonStore(
                self._num_persons,
                num_ticks,
                self._static_mapping,
                self._dynamic_mapping,
                self._event_mapping,
                npz_file=f.name,
            )

        all_prop_names = (
            self._static_mapping.property_mappings.keys()
            | self._dynamic_mapping.property_mappings.keys()
            | self._event_mapping.property_mappings.keys()
        )
        orig_pop = original_store.get_population_at(t=0)
        load_pop = loaded_store.get_population_at(t=0)
        for p, q in zip(orig_pop, load_pop):
            for n in all_prop_names:
                p_val = getattr(p.current, n)
                q_val = getattr(q.current, n)
                self.assertEqual(p_val, q_val)

    def test_npz_file_object_loads_identical_data(self):
        num_ticks = 1
        original_store = NumpyPersonStore(
            self._num_persons,
            num_ticks,
            self._static_mapping,
            self._dynamic_mapping,
            self._event_mapping,
            initial_person_records=self._person_records,
        )
        with TemporaryFile() as f:
            original_store.save_to_file(f)
            f.flush()
            f.seek(0)

            loaded_store = NumpyPersonStore(
                self._num_persons,
                num_ticks,
                self._static_mapping,
                self._dynamic_mapping,
                self._event_mapping,
                npz_file=f,
            )

        all_prop_names = (
            self._static_mapping.property_mappings.keys()
            | self._dynamic_mapping.property_mappings.keys()
            | self._event_mapping.property_mappings.keys()
        )
        orig_pop = original_store.get_population_at(t=0)
        load_pop = loaded_store.get_population_at(t=0)
        for p, q in zip(orig_pop, load_pop):
            for n in all_prop_names:
                p_val = getattr(p.current, n)
                q_val = getattr(q.current, n)
                self.assertEqual(p_val, q_val)

    def test_initial_records_and_npz_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        expected_err_msg = "Can load either initial records or an `.npz` file, but not both"

        with self.assertRaises(expected_err_type) as c:
            with TemporaryFile() as f:
                NumpyPersonStore(
                    self._num_persons,
                    num_ticks,
                    self._static_mapping,
                    self._dynamic_mapping,
                    self._event_mapping,
                    initial_person_records=self._person_records,
                    npz_file=f,
                )
        actual_err_msg = c.exception.args[0]
        self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_missing_arrays_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        all_arrays = {
            "static": np.empty((self._num_persons,), dtype=self._static_mapping.dtype),
            "dynamic": np.empty(
                (self._num_persons, num_ticks + 1), dtype=self._dynamic_mapping.dtype
            ),
            "event": np.empty((self._num_persons, num_ticks + 1), dtype=self._event_mapping.dtype),
        }
        arrays_to_remove = [
            {"static"},
            {"dynamic"},
            {"event"},
            {"static", "dynamic"},
            {"static", "event"},
            {"dynamic", "event"},
            {"static", "dynamic", "event"},
        ]
        for missing_arrays in arrays_to_remove:
            arrays_to_write = {k: v for k, v in all_arrays.items() if k not in missing_arrays}
            expected_err_msg = f"Required arrays not found: {sorted(missing_arrays)}"

            with TemporaryFile() as f:
                np.savez_compressed(f, **arrays_to_write)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_static_shape_dtype_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_shape = (self._num_persons,)
        required_dtype = self._static_mapping.dtype
        mismatch_shapes = [
            (self._num_persons + 1,),
            (self._num_persons - 1,),
            (0,),
            (1,),
            (self._num_persons, num_ticks + 1),
        ]
        mismatch_dtypes = [
            np.dtype("float32"),
            np.dtype([("i", "i4")]),
            self._dynamic_mapping.dtype,
            self._event_mapping.dtype,
        ]
        for static_shape, static_dtype in itertools.product(mismatch_shapes, mismatch_dtypes):
            array_data = {
                "static": np.empty(static_shape, dtype=static_dtype),
                "dynamic": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._dynamic_mapping.dtype
                ),
                "event": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._event_mapping.dtype
                ),
            }
            expected_err_msg = (
                "Static array does not have the required dtype"
                f" ({static_dtype} != {required_dtype}) nor the required shape"
                f" ({static_shape} != {required_shape})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_static_dtype_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_dtype = self._static_mapping.dtype
        mismatch_dtypes = [
            np.dtype("float32"),
            np.dtype([("i", "i4")]),
            self._dynamic_mapping.dtype,
            self._event_mapping.dtype,
        ]
        for static_dtype in mismatch_dtypes:
            array_data = {
                "static": np.empty((self._num_persons,), dtype=static_dtype),
                "dynamic": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._dynamic_mapping.dtype
                ),
                "event": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._event_mapping.dtype
                ),
            }
            expected_err_msg = (
                "Static array does not have the required dtype"
                f" ({static_dtype} != {required_dtype})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_static_shape_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_shape = (self._num_persons,)
        mismatch_shapes = [
            (self._num_persons + 1,),
            (self._num_persons - 1,),
            (0,),
            (1,),
            (self._num_persons, num_ticks + 1),
        ]
        for static_shape in mismatch_shapes:
            array_data = {
                "static": np.empty(static_shape, dtype=self._static_mapping.dtype),
                "dynamic": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._dynamic_mapping.dtype
                ),
                "event": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._event_mapping.dtype
                ),
            }
            expected_err_msg = (
                "Static array does not have the required shape"
                f" ({static_shape} != {required_shape})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_dynamic_shape_dtype_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_shape = (self._num_persons, num_ticks + 1)
        required_dtype = self._dynamic_mapping.dtype
        mismatch_shapes = [
            (self._num_persons + 1, required_shape[1]),
            (self._num_persons - 1, required_shape[1]),
            (self._num_persons, required_shape[1] + 1),
            (self._num_persons, required_shape[1] - 1),
            (self._num_persons + 1, required_shape[1] + 1),
            (self._num_persons + 1, required_shape[1] - 1),
            (self._num_persons - 1, required_shape[1] + 1),
            (self._num_persons - 1, required_shape[1] - 1),
            (self._num_persons,),
            (0,),
            (0, 0),
            (1,),
            (1, 1),
        ]
        mismatch_dtypes = [
            np.dtype("float32"),
            np.dtype([("i", "i4")]),
            self._static_mapping.dtype,
            self._event_mapping.dtype,
        ]
        for dynamic_shape, dynamic_dtype in itertools.product(mismatch_shapes, mismatch_dtypes):
            array_data = {
                "static": np.empty((self._num_persons,), dtype=self._static_mapping.dtype),
                "dynamic": np.empty(dynamic_shape, dtype=dynamic_dtype),
                "event": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._event_mapping.dtype
                ),
            }
            expected_err_msg = (
                "Dynamic array does not have the required dtype"
                f" ({dynamic_dtype} != {required_dtype}) nor the required shape"
                f" ({dynamic_shape} != {required_shape})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_dynamic_dtype_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_dtype = self._dynamic_mapping.dtype
        mismatch_dtypes = [
            np.dtype("float32"),
            np.dtype([("i", "i4")]),
            self._static_mapping.dtype,
            self._event_mapping.dtype,
        ]
        for dynamic_dtype in mismatch_dtypes:
            array_data = {
                "static": np.empty((self._num_persons,), dtype=self._static_mapping.dtype),
                "dynamic": np.empty((self._num_persons, num_ticks + 1), dtype=dynamic_dtype),
                "event": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._event_mapping.dtype
                ),
            }
            expected_err_msg = (
                "Dynamic array does not have the required dtype"
                f" ({dynamic_dtype} != {required_dtype})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_dynamic_shape_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_shape = (self._num_persons, num_ticks + 1)
        mismatch_shapes = [
            (self._num_persons, required_shape[1] + 1),
            (self._num_persons, required_shape[1] - 1),
            (self._num_persons + 1, required_shape[1]),
            (self._num_persons - 1, required_shape[1]),
            (self._num_persons + 1, required_shape[1] + 1),
            (self._num_persons + 1, required_shape[1] - 1),
            (self._num_persons - 1, required_shape[1] + 1),
            (self._num_persons - 1, required_shape[1] - 1),
            (self._num_persons,),
            (0,),
            (0, 0),
            (1,),
            (1, 1),
        ]
        for dynamic_shape in mismatch_shapes:
            array_data = {
                "static": np.empty((self._num_persons,), dtype=self._static_mapping.dtype),
                "dynamic": np.empty(dynamic_shape, dtype=self._dynamic_mapping.dtype),
                "event": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._event_mapping.dtype
                ),
            }
            expected_err_msg = (
                "Dynamic array does not have the required shape"
                f" ({dynamic_shape} != {required_shape})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_event_shape_dtype_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_shape = (self._num_persons, num_ticks + 1)
        required_dtype = self._event_mapping.dtype
        mismatch_shapes = [
            (self._num_persons + 1, required_shape[1]),
            (self._num_persons - 1, required_shape[1]),
            (self._num_persons, required_shape[1] + 1),
            (self._num_persons, required_shape[1] - 1),
            (self._num_persons + 1, required_shape[1] + 1),
            (self._num_persons + 1, required_shape[1] - 1),
            (self._num_persons - 1, required_shape[1] + 1),
            (self._num_persons - 1, required_shape[1] - 1),
            (self._num_persons,),
            (0,),
            (0, 0),
            (1,),
            (1, 1),
        ]
        mismatch_dtypes = [
            np.dtype("float32"),
            np.dtype([("i", "i4")]),
            self._static_mapping.dtype,
            self._dynamic_mapping.dtype,
        ]
        for event_shape, event_dtype in itertools.product(mismatch_shapes, mismatch_dtypes):
            array_data = {
                "static": np.empty((self._num_persons,), dtype=self._static_mapping.dtype),
                "dynamic": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._dynamic_mapping.dtype
                ),
                "event": np.empty(event_shape, dtype=event_dtype),
            }
            expected_err_msg = (
                "Event array does not have the required dtype"
                f" ({event_dtype} != {required_dtype}) nor the required shape"
                f" ({event_shape} != {required_shape})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_event_dtype_mismatch_raises_error(self):
        num_ticks = 1
        expected_err_type = ValueError
        required_dtype = self._event_mapping.dtype
        mismatch_dtypes = [
            np.dtype("float32"),
            np.dtype([("i", "i4")]),
            self._static_mapping.dtype,
            self._dynamic_mapping.dtype,
        ]
        for event_dtype in mismatch_dtypes:
            array_data = {
                "static": np.empty((self._num_persons,), dtype=self._static_mapping.dtype),
                "dynamic": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._dynamic_mapping.dtype
                ),
                "event": np.empty((self._num_persons, num_ticks + 1), dtype=event_dtype),
            }
            expected_err_msg = (
                "Event array does not have the required dtype"
                f" ({event_dtype} != {required_dtype})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)

    def test_npz_file_event_shape_mismatch_raises_error(self):
        num_ticks = 1
        required_shape = (self._num_persons, num_ticks + 1)
        expected_err_type = ValueError
        mismatch_shapes = [
            (self._num_persons, required_shape[1] + 1),
            (self._num_persons, required_shape[1] - 1),
            (self._num_persons + 1, required_shape[1]),
            (self._num_persons - 1, required_shape[1]),
            (self._num_persons + 1, required_shape[1] + 1),
            (self._num_persons + 1, required_shape[1] - 1),
            (self._num_persons - 1, required_shape[1] + 1),
            (self._num_persons - 1, required_shape[1] - 1),
            (self._num_persons,),
            (0,),
            (0, 0),
            (1,),
            (1, 1),
        ]
        for event_shape in mismatch_shapes:
            array_data = {
                "static": np.empty((self._num_persons,), dtype=self._static_mapping.dtype),
                "dynamic": np.empty(
                    (self._num_persons, num_ticks + 1), dtype=self._dynamic_mapping.dtype
                ),
                "event": np.empty(event_shape, dtype=self._event_mapping.dtype),
            }
            expected_err_msg = (
                "Event array does not have the required shape"
                f" ({event_shape} != {required_shape})"
            )

            with TemporaryFile() as f:
                np.savez_compressed(f, **array_data)
                f.flush()
                f.seek(0)

                with self.assertRaises(expected_err_type) as c:
                    NumpyPersonStore(
                        self._num_persons,
                        num_ticks,
                        self._static_mapping,
                        self._dynamic_mapping,
                        self._event_mapping,
                        npz_file=f,
                    )
                actual_err_msg = c.exception.args[0]
                self.assertEqual(expected_err_msg, actual_err_msg)
