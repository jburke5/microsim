class NumpyPopulationRecordProxy:
    def __init__(self, static_rows, dynamic_rows, event_rows):
        self._static_rows = static_rows
        self._dynamic_rows = dynamic_rows
        self._event_rows = event_rows
