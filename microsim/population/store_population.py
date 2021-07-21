class StorePopulation:
    """
    Population that uses a PersonStore to store its people.

    Primarily intended for storing Person data in Numpy ndarrays for memory
    efficiency while still having Person-like objects for advancing, updating,
    and analyzing.
    """

    def __init__(self, person_store):
        self._person_store = person_store
        self._current_tick = 0

    @property
    def person_store(self):
        return self._person_store

    @property
    def current_tick(self):
        return self._current_tick

    def advance(self, num_ticks=1):
        """Advance population by a given number of ticks (default: 1)."""
        for tick_index in range(num_ticks):
            t = self._current_tick + tick_index + 1
            advance_record_window = self._person_store.get_population_advance_record_window(
                t,
                condition=lambda p: p.alive,
            )
            current_population_record, next_population_record = advance_record_window

            # TODO: the rest of `advance` goes here

            self._current_tick = t
        raise NotImplementedError()
