class NumpyPopulationRecordProxy:
    def __init__(
        self,
        static_rows,
        dynamic_rows,
        event_rows,
        static_converter,
        dynamic_converter,
        event_converter,
    ):
        self._static_rows = static_rows
        self._dynamic_rows = dynamic_rows
        self._event_rows = event_rows
        self._static_converter = static_converter
        self._dynamic_converter = dynamic_converter
        self._event_converter = event_converter
