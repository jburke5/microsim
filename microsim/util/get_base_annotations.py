def get_base_annotations(cls):
    """Returns dict of class attr annotations from all base classes."""

    annotations = {}
    for base_class in reversed(cls.__mro__):
        try:
            annotations.update(base_class.__annotations__)
        except AttributeError:
            pass  # okay if not all classes have __annotations__
    return annotations
