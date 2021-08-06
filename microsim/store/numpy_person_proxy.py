class NumpyPersonProxy:
    def __init__(self, current_proxy, prev_proxies):
        self._current_proxy = current_proxy
        self._prev_proxies = prev_proxies

    @property
    def current(self):
        return self._current_proxy

    @property
    def previous(self):
        return self._prev_proxies[-1]

    def all_previous(self):
        return list(self._prev_proxies)
