class Person:
    """Person is using risk factors and demographics based off NHANES"""

    def __init__(self, age, gender, race_ethnicity, sbp, dbp, a1c, hdl, chol):
        self._gender = gender
        self._race_ethnicity = race_ethnicity

        self._age = [age]
        self._sbp = [sbp]
        self._dbp = [dbp]
        self._a1c = [a1c]
        self._hdl = [hdl]
        self._chol = [chol]
