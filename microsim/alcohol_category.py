from enum import IntEnum
import pandas as pd
import numpy as np


class AlcoholCategory(IntEnum):
    NONE = 0
    ONETOSIX = 1
    SEVENTOTHIRTEEN = 2
    FOURTEENORMORE = 3

    @staticmethod
    def get_category_for_consumption(drinks_per_week):
        return AlcoholCategory(pd.cut([drinks_per_week], [-1, 0, 6, 13, np.Infinity]).codes[0])
