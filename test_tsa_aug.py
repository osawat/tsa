import unittest
import numpy as np

import tsa_aug


class TestTsaAug(unittest.TestCase):
    def setUp(self) -> None:
        n = 100
        x_sin = np.sin(3 * np.pi * np.arange(n) / n).reshape(-1, 1)
        x_cos = np.cos(2 * np.pi * np.arange(n) / n).reshape(-1, 1)
        self.X = np.concatenate([x_sin, x_cos], axis=1)

    def assertEqualNdArray(self, x1, x2):
        self.assertTrue(np.allclose(x1, x2, rtol=1e-05, atol=1e-08))


class TestTsJitter(TestTsaAug):

    def test_jitter_cls(self):
        x_ans = tsa_aug.jitter(self.X, random_seed=101)
        trans_cls = tsa_aug.TsJitter(random_seed=101)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))


class TestTsScalar(TestTsaAug):

    def test_fit_transform(self):
        x_ans = tsa_aug.scaling(self.X, random_seed=101)
        trans_cls = tsa_aug.TsScalar(random_seed=101)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))


class TestTsMagnitudeWarp(TestTsaAug):

    def test_fit_transform(self):
        x_ans = tsa_aug.magnitude_warp(self.X, random_seed=101)
        trans_cls = tsa_aug.TsMagnitudeWarp(random_seed=101)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))

        x_ans = tsa_aug.magnitude_warp(self.X, individual=True, random_seed=101)
        trans_cls = tsa_aug.TsMagnitudeWarp(individual=True, random_seed=101)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))


class TestTsTimeWarp(TestTsaAug):

    def test_fit_transform(self):
        x_ans = tsa_aug.time_warp(self.X, random_seed=101)
        trans_cls = tsa_aug.TsTimeWarp(random_seed=101)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))

        x_ans = tsa_aug.time_warp(self.X, individual=True, random_seed=101)
        trans_cls = tsa_aug.TsTimeWarp(individual=True, random_seed=101)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))


class TestStretch(TestTsaAug):

    def test_fit_transform(self):
        x_ans = tsa_aug.time_stretch(self.X, scale=0.5)
        trans_cls = tsa_aug.TsTimeStretch(scale=0.5)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_almost_equal(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))


class TestTsPartialTimeStretch(TestTsaAug):

    def test_fit_transform(self):
        stretches = [(0.2, 0.4, 2), (0.5, 0.7, 3)]
        x_ans = tsa_aug.partial_time_stretch(self.X, stretches)
        trans_cls = tsa_aug.TsPartialTimeStretch(stretches)
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_almost_equal(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))


class TestResampling(TestTsaAug):

    def test_fit_transform(self):
        x_ans = tsa_aug.resampling(self.X, src_period=10, dst_period=5)
        trans_cls = tsa_aug.TsResampling(src_period=10, dst_period=5)
        self.assertTrue((x_ans == trans_cls.fit_transform(self.X)).all())
        self.assertEqualNdArray(x_ans, trans_cls.fit_transform(self.X))
        np.testing.assert_array_equal(x_ans, trans_cls.fit_transform(self.X))


if __name__ == '__main__':
    unittest.main()
