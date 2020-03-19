import unittest

from microsim.alcohol_category import AlcoholCategory


class TestAlcoholCategories(unittest.TestCase):

    def test_cut_points(self):
        self.assertEqual(AlcoholCategory.NONE, AlcoholCategory.get_category_for_consumption(0))
        self.assertEqual(AlcoholCategory.ONETOSIX, AlcoholCategory.get_category_for_consumption(1))
        self.assertEqual(AlcoholCategory.ONETOSIX, AlcoholCategory.get_category_for_consumption(6))
        self.assertEqual(AlcoholCategory.SEVENTOTHIRTEEN, AlcoholCategory.get_category_for_consumption(7))
        self.assertEqual(AlcoholCategory.SEVENTOTHIRTEEN, AlcoholCategory.get_category_for_consumption(13))
        self.assertEqual(AlcoholCategory.FOURTEENORMORE, AlcoholCategory.get_category_for_consumption(14))
        self.assertEqual(AlcoholCategory.FOURTEENORMORE, AlcoholCategory.get_category_for_consumption(100))


if __name__ == "__main__":
    unittest.main()
