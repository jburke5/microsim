from enum import IntEnum


class AlcoholCategory(IntEnum):
    """One to six refers here to standardized drinks per week, so the unit here is a drink"""

    NONE = 0
    ONETOSIX = 1
    SEVENTOTHIRTEEN = 2
    FOURTEENORMORE = 3

    @staticmethod
    def get_category_for_consumption(drinks_per_week):
        # same bins as pd.cut([-1, 0, 6, 13, inf]), right-inclusive, without its per-call cost
        if not drinks_per_week > -1:  # NaN or <= -1, which pd.cut left without a category
            raise ValueError(f"{drinks_per_week} drinks per week has no AlcoholCategory")
        if drinks_per_week <= 0:
            return AlcoholCategory.NONE
        if drinks_per_week <= 6:
            return AlcoholCategory.ONETOSIX
        if drinks_per_week <= 13:
            return AlcoholCategory.SEVENTOTHIRTEEN
        return AlcoholCategory.FOURTEENORMORE
