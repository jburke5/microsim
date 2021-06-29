def inherit_annotations(cls):
    """
    Inherits class attribute annotations from all base classes.

    Convenience function for creating dataclasses from Protocols that define
    the class attributes but do not implement them.
    """

    annotations = {}
    for base_class in reversed(cls.__mro__):
        annotations.update(base_class.__annotations__)
    cls.__annotations__ = annotations
    return cls
