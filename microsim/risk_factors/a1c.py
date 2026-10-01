# Using this paper...glucose and a1c are highly related
# Nathan, D. M., Kuenen, J., Borg, R., Zheng, H., Schoenfeld, D., Heine, R. J., for the A1c-Derived Average Glucose (ADAG) Study Group. (2008). Translating the A1C Assay Into Estimated Average Glucose Values. Diabetes Care, 31(8), 1473–1478.
# so, will use their formula + a draw from residual distribution fo same moddel in NHANES (which has very simnilar coefficients)


def convert_fasting_glucose_to_a1c(glucose):
    return (glucose + 46.7) / 28.7


def convert_a1c_to_fasting_glucose(a1c):
    return 28.7 * a1c - 46.7
