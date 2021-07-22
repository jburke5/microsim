class NumpyFieldProxy:
    """Descriptor that proxies a single field of a row of a Numpy array."""

    def __init__(self, row_attr_name, np_field_name, to_np, from_np):
        self._row_attr_name = row_attr_name
        self._np_field_name = np_field_name
        self._to_np = to_np
        self._from_np = from_np

    def __get__(self, instance, owner=None):
        if instance is None:
            raise AttributeError("NumpyFieldProxy not defined for class attributes")

        row = getattr(instance, self._row_attr_name)
        np_value = row[self._np_field_name]
        value = self._from_np(np_value)
        return value

    def __set__(self, instance, value):
        row = getattr(instance, self._row_attr_name)
        np_value = self._to_np(value)
        row[self._np_field_name] = np_value
