from typing import get_type_hints


def get_base_annotations(cls):
    """Returns dict of class attr annotations from all base classes."""

    annotations = {}
    for base_class in reversed(cls.__mro__):
        try:
            base_class_annotations = get_type_hints(base_class)
            annotations.update(base_class_annotations)
        except AttributeError:
            pass  # okay if not all classes have __annotations__
    return annotations
