import unittest

from agents.config import USD_KRW_FIXED_RATE
from agents.stubs import _number_in_text, _scaled_money


class MoneyScalingTest(unittest.TestCase):
    def test_scale_word_inside_value(self):
        self.assertEqual(_scaled_money("1,228억", "달러", "USD"), 122_800_000_000)
        self.assertEqual(_scaled_money("610억", "원", "KRW"), 61_000_000_000)

    def test_scale_word_inside_unit(self):
        self.assertEqual(_scaled_money(71.3, "billion USD", "USD"), 71_300_000_000)
        self.assertEqual(_scaled_money(3, "조원", "KRW"), 3_000_000_000_000)

    def test_currency_conversion_uses_config_rate(self):
        self.assertAlmostEqual(_scaled_money(1_000_000, "USD", "KRW"), 1_000_000 * USD_KRW_FIXED_RATE)

    def test_unknown_currency_is_missing(self):
        self.assertIsNone(_scaled_money(100, "EUR", "USD"))

    def test_number_must_appear_in_quote(self):
        self.assertTrue(_number_in_text("1,228억", "2026년 시장 규모는 1,228억 달러로 예상된다"))
        self.assertFalse(_number_in_text(52, "2026년 시장 규모는 1,228억 달러로 예상된다"))


if __name__ == "__main__":
    unittest.main()
