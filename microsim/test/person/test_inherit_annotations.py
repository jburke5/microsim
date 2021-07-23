from microsim.util.inherit_annotations import inherit_annotations
from unittest import TestCase


class Baseless:
    pass


class DimX:
    x: int


class DimY:
    y: int


class DimZ:
    z: int


class DimZReal:
    z: float


class NaturalNumberLine(DimX):
    pass


class Point3D(DimX, DimY, DimZ):
    pass


class Point3DNatX(NaturalNumberLine, DimY, DimZ):
    pass


class Point3DStillIntZ(Point3D, DimZReal):
    pass


class Point3DRealZ(DimZReal, Point3D):
    pass


class TestInheritAnnotations(TestCase):
    def setUp(self):
        test_classes_with_annotations = [
            DimX,
            DimY,
            DimZ,
            DimZReal,
            NaturalNumberLine,
            Point3D,
            Point3DNatX,
            Point3DStillIntZ,
            Point3DRealZ,
        ]
        self._original_annotations = [
            (cls, {**cls.__annotations__}) for cls in test_classes_with_annotations
        ]

    def tearDown(self):
        # need to reset __annotations__ of all used classes after tests
        try:
            delattr(Baseless, "__annotations__")
        except AttributeError:
            pass

        for cls, annotations in self._original_annotations:
            cls.__annotations__ = {**annotations}

    def test_no_bases_only_class_attrs(self):
        cls = DimX
        expected_annotations = {"x": int}

        decorated_cls = inherit_annotations(cls)

        self.assertDictEqual(expected_annotations, decorated_cls.__annotations__)

    def test_no_bases_no_attrs_annotations_not_set(self):
        cls = Baseless
        expected_annotations = {}

        decorated_cls = inherit_annotations(cls)

        self.assertEqual(expected_annotations, decorated_cls.__annotations__)

    def test_single_base_inherits_all(self):
        cls = NaturalNumberLine
        expected_annotations = {"x": int}

        decorated_cls = inherit_annotations(cls)

        self.assertDictEqual(expected_annotations, decorated_cls.__annotations__)

    def test_multiple_bases_inherits_all(self):
        cls = Point3D
        expected_annotations = {"x": int, "y": int, "z": int}

        decorated_cls = inherit_annotations(cls)

        self.assertDictEqual(expected_annotations, decorated_cls.__annotations__)

    def test_nested_bases_still_inherits_all(self):
        cls = Point3DNatX
        expected_annotations = {"x": int, "y": int, "z": int}

        decorated_cls = inherit_annotations(cls)

        self.assertDictEqual(expected_annotations, decorated_cls.__annotations__)

    def test_multiple_bases_last_base_overriden(self):
        cls = Point3DStillIntZ
        expected_annotations = {"x": int, "y": int, "z": int}

        decorated_cls = inherit_annotations(cls)

        self.assertDictEqual(expected_annotations, decorated_cls.__annotations__)

    def test_multiple_bases_first_base_overrides(self):
        cls = Point3DRealZ
        expected_annotations = {"x": int, "y": int, "z": float}

        decorated_cls = inherit_annotations(cls)

        self.assertDictEqual(expected_annotations, decorated_cls.__annotations__)
