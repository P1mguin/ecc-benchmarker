import unittest

from ecc.curve import (
    ShortWeierstrassCurves,
    MontgomeryCurves,
    Curve25519,
    ShortWeierstrassBinaryCurve,
    Point
)

CURVES = ShortWeierstrassCurves + MontgomeryCurves


class PointAndCurveTestCase(unittest.TestCase):
    def test_operator(self):
        for curve in CURVES:
            with self.subTest(curve=curve):
                P = curve.G
                self.assertEqual(P + P, 2 * P)
                self.assertEqual(P - P, curve.INF)
                self.assertEqual(P + (-P), curve.INF)
                self.assertEqual(P + P + P + P + P, P * 5)
                self.assertEqual(-P - P - P - P - P, -5 * P)
                self.assertEqual(P - 2 * P, -P)
                self.assertEqual(20 * P + 4 * P, 10 * P + 14 * P)
                self.assertEqual(curve.INF + 10 * P, 10 * P)
                self.assertEqual(curve.INF - 3 * P, -3 * P)
                self.assertEqual(curve.INF + curve.INF, curve.INF)
                self.assertEqual(0 * P, curve.INF)
                self.assertEqual(1000 * curve.INF, curve.INF)

    def test_double_points_y_equals_to_0(self):
        P = Point(x=0, y=0, curve=Curve25519)
        self.assertEqual(P + P, Curve25519.INF)
        self.assertEqual(2 * P, Curve25519.INF)
        self.assertEqual(-2 * P, Curve25519.INF)


class BinaryCurveTest(unittest.TestCase):
    def test_operator(self):
        curve = ShortWeierstrassBinaryCurve(
            name="test",
            a=0,
            b=1,
            p=4,
            n=10,
            G_x=1,
            G_y=1
        )
        with self.subTest(curve=curve):
            # Base points for testing
            P = curve.G
            Q = Point(3, 1, curve)
            print(P, Q)

            # Testing negation
            self.assertEqual((-Q).y, 0)
            self.assertEqual((-P).y, 2)

            # Testing addition
            self.assertEqual((P + Q).x, 0)
            self.assertEqual((P + Q).y, 1)

            # Testing scalar multiplication
            self.assertEqual(2 * P, P + P)

            # Testing finding a point on the curve
            self.assertEqual(curve.compute_y(1), 1)
            self.assertEqual(curve._compute_up_y(1), 1)
            self.assertEqual(curve._compute_down_y(1), 2)

            # Testing is on curve
            self.assertTrue(curve.is_on_curve(P))

    def test_base_points(self):
        for curve in ShortWeierstrassPrimeNormalCurves + ShortWeierstrassPrimePolynomialCurves + KoblitzPolynomialCurves + KoblitzNormalCurves:
            with self.subTest(curve=curve):
                print(curve, curve.is_on_curve(curve.G))
                self.assertTrue(curve.is_on_curve(curve.G))