from typing import get_type_hints


def inherit_annotations(cls):
    """
    Inherits class attribute annotations from all base classes.

    Convenience function for creating dataclasses from Protocols that define
    the class attributes but do not implement them.
    """

    cls.__annotations__ = get_type_hints(cls)
    return cls
