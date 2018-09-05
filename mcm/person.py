class Person:
    def __init__(self, initial_risk_factors):
        self._risk_factor_at_time = [
            initial_risk_factors,
        ]

    def get_current_risk_factors(self):
        return self._risk_factor_at_time[-1]
