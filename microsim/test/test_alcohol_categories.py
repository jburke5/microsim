import unittest

import numpy as np
import pandas as pd

from microsim.risk_factors.alcohol_category import AlcoholCategory


class TestAlcoholCategories(unittest.TestCase):
    def test_cut_points(self):
        self.assertEqual(AlcoholCategory.NONE, AlcoholCategory.get_category_for_consumption(0))
        self.assertEqual(AlcoholCategory.ONETOSIX, AlcoholCategory.get_category_for_consumption(1))
        self.assertEqual(AlcoholCategory.ONETOSIX, AlcoholCategory.get_category_for_consumption(6))
        self.assertEqual(
            AlcoholCategory.SEVENTOTHIRTEEN, AlcoholCategory.get_category_for_consumption(7)
        )
        self.assertEqual(
            AlcoholCategory.SEVENTOTHIRTEEN, AlcoholCategory.get_category_for_consumption(13)
        )
        self.assertEqual(
            AlcoholCategory.FOURTEENORMORE, AlcoholCategory.get_category_for_consumption(14)
        )
        self.assertEqual(
            AlcoholCategory.FOURTEENORMORE, AlcoholCategory.get_category_for_consumption(100)
        )

    def test_matches_pd_cut(self):
        # the pd.cut the comparisons replaced
        for drinks in [-0.5, 0, 0.01, 0.5, 1, 5.99, 6, 6.01, 7, 12.99, 13, 13.01, 14, 100]:
            expected = AlcoholCategory(pd.cut([drinks], [-1, 0, 6, 13, np.inf]).codes[0])
            self.assertEqual(expected, AlcoholCategory.get_category_for_consumption(drinks))

    def test_out_of_range_raises(self):
        for drinks in [-1, -5, np.nan]:
            with self.assertRaises(ValueError):
                AlcoholCategory.get_category_for_consumption(drinks)


if __name__ == "__main__":
    unittest.main()
