class NumpyPersonProxy:
    def __init__(self, next_record, cur_prev_records):
        self._next_record = next_record
        self._cur_prev_records = list(cur_prev_records)

    @property
    def next(self):
        return self._next_record

    @property
    def current(self):
        return self._cur_prev_records[-1]

    @property
    def current_and_previous(self):
        return self._cur_prev_records[:]
