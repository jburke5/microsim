from microsim.util.get_base_annotations import get_base_annotations


def inherit_annotations(cls):
    """
    Inherits class attribute annotations from all base classes.

    Convenience function for creating dataclasses from Protocols that define
    the class attributes but do not implement them.
    """

    cls.__annotations__ = get_base_annotations(cls)
    return cls
