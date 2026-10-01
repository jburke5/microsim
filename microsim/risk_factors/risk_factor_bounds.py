from microsim.risk_factors.risk_factor import DynamicRiskFactorsType

class RiskFactorBounds:
    """Static prespecified bounds for dynamic risk factors."""

    #bounds based on NHANES data from 1999 to 2017 (all data), 0.9*nhanesMin, 1.1*nhanesMax
    #age is an exception, bounds set manually
    _lowerBoundsAdult = {
                     DynamicRiskFactorsType.SBP.value: 58.20,
                     DynamicRiskFactorsType.DBP.value: 36. ,
                     DynamicRiskFactorsType.CREATININE.value: 0.090,
                     DynamicRiskFactorsType.WAIST.value: 49.95,
                     DynamicRiskFactorsType.LDL.value: 8.10,
                     DynamicRiskFactorsType.A1C.value: 1.80,
                     DynamicRiskFactorsType.TRIG.value: 9.,
                     DynamicRiskFactorsType.BMI.value: 10.836,
                     DynamicRiskFactorsType.HDL.value: 5.4,
                     DynamicRiskFactorsType.AGE.value: 18,
                     DynamicRiskFactorsType.TOT_CHOL.value: 53.1}
    _upperBoundsAdult = {
                     DynamicRiskFactorsType.SBP.value: 297.,
                     DynamicRiskFactorsType.DBP.value: 152.53,
                     DynamicRiskFactorsType.CREATININE.value: 19.58,
                     DynamicRiskFactorsType.WAIST.value: 196.9,
                     DynamicRiskFactorsType.LDL.value: 691.9,
                     DynamicRiskFactorsType.A1C.value: 20.68,
                     DynamicRiskFactorsType.TRIG.value: 4656.3,
                     DynamicRiskFactorsType.BMI.value: 143.23,
                     DynamicRiskFactorsType.HDL.value: 248.6,
                     DynamicRiskFactorsType.AGE.value: 130,
                     DynamicRiskFactorsType.TOT_CHOL.value: 894.3}
    _lowerBoundsChild = {
                     DynamicRiskFactorsType.SBP.value: 66.6,
                     DynamicRiskFactorsType.DBP.value: 36. ,
                     DynamicRiskFactorsType.CREATININE.value: 0.126,
                     DynamicRiskFactorsType.WAIST.value: 34.02,
                     DynamicRiskFactorsType.LDL.value: 8.10,
                     DynamicRiskFactorsType.A1C.value: 3.42,
                     DynamicRiskFactorsType.TRIG.value: 9.,
                     DynamicRiskFactorsType.BMI.value: 10.35,
                     DynamicRiskFactorsType.HDL.value: 9.9,
                     DynamicRiskFactorsType.AGE.value: 0.,
                     DynamicRiskFactorsType.TOT_CHOL.value: 59.4}
    _upperBoundsChild = {
                     DynamicRiskFactorsType.SBP.value: 190.3,
                     DynamicRiskFactorsType.DBP.value: 114.4,
                     DynamicRiskFactorsType.CREATININE.value: 13.728,
                     DynamicRiskFactorsType.WAIST.value: 183.92,
                     DynamicRiskFactorsType.LDL.value: 282.7,
                     DynamicRiskFactorsType.A1C.value: 17.16,
                     DynamicRiskFactorsType.TRIG.value: 1718.2,
                     DynamicRiskFactorsType.BMI.value: 68.288,
                     DynamicRiskFactorsType.HDL.value: 196.9,
                     DynamicRiskFactorsType.AGE.value: 17,
                     DynamicRiskFactorsType.TOT_CHOL.value: 484.}
    _upperBounds = {"adult": _upperBoundsAdult,
                    "child": _upperBoundsChild}
    _lowerBounds = {"adult": _lowerBoundsAdult,
                    "child": _lowerBoundsChild}

    @classmethod
    def apply(cls, varName, varValue, adult=True):
        """
        Ensures that risk factor are within static prespecified bounds.

        Other algorithms might be needed in the future to avoid pooling in the tails,
        if there are many extreme risk factor results.
        """
        person = "adult" if adult else "child"
        if varName in cls._upperBounds[person]:
            upperBound = cls._upperBounds[person][varName]
            varValue = varValue if varValue < upperBound else upperBound
        if varName in cls._lowerBounds[person]:
            lowerBound = cls._lowerBounds[person][varName]
            varValue = varValue if varValue > lowerBound else lowerBound
        return varValue

    @classmethod
    def apply_to_person(cls, varName, varValue, person):
        """Applies bounds choosing the adult/child tables from the person's age.

        Age itself is judged by the proposed next value, so a 17-year-old
        advancing to 18 is not clamped back by the child upper bound."""
        age = varValue if varName == DynamicRiskFactorsType.AGE.value else person._age[-1]
        return cls.apply(varName, varValue, adult=age >= 18)
