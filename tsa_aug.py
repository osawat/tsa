import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, TransformerMixin


def jitter(xs, sigma=0.05, random_seed=None):
    """
    :param xs: Data:numpy.array [-1, x_dim]
    :param sigma:
    :param random_seed:
    :return: X with noise numpy:array[-1, n_dim]
    jitterは各々のデータに対して個別のノイズを加えます
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    noise = np.random.normal(loc=0, scale=sigma, size=xs.shape)
    return xs + noise


class TsJitter(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=0.05, noise=None, random_seed=None):
        self.sigma = sigma
        self.noise = noise
        self.random_seed = random_seed

    def fit(self, xs):
        if self.noise is None:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            self.noise = np.random.normal(loc=0, scale=self.sigma, size=xs.shape)
        return self

    def transform(self, xs):
        return xs + self.noise


def scaling(xs, sigma=0.1, individual=False, random_seed=None):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param sigma:
    :param individual: bool 個別にスケーリングするか否か
    :param random_seed:
    :return: numpy.array [-1, n_dim]
    scalingはデータ系列毎にスケーリング係数を乗じます
    """
    x_size, x_dim = xs.shape[0], xs.shape[1]
    if random_seed is not None:
        np.random.seed(random_seed)
    if individual:
        scaling_factor = np.random.normal(loc=1, scale=sigma, size=(1, x_dim))  # (1, x_dim)
        return xs * scaling_factor
    else:
        scaling_factor = np.random.normal(loc=1, scale=sigma, size=(1, 1))  # (1, x_dim)
        return xs * scaling_factor


class TsScalar(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=0.1, scaling_factor=None, individual=False, random_seed=None):
        self.sigma = sigma
        self.individual = individual
        self.scaling_factor = scaling_factor
        self.random_seed = random_seed

    def fit(self, xs):
        x_size, x_dim = xs.shape[0], xs.shape[1]
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if self.individual:
            if self.scaling_factor is None:
                self.scaling_factor = np.random.normal(loc=1, scale=self.sigma, size=(1, x_dim))  # (1, x_dim)
        else:
            if self.scaling_factor is None:
                self.scaling_factor = np.random.normal(loc=1, scale=self.sigma, size=(1, 1))  # (1, x_dim)
        return self

    def transform(self, xs):
        return xs * self.scaling_factor


def generate_random_curves(xs, sigma=0.2, n_knots=4, random_seed=None):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param sigma:
    :param n_knots: 折り曲げの回数
    :param random_seed:
    :return: numpy.array [-1, n_dim]
    データ系列毎にデータ全体を１を中心にn_knotsだけ折り曲げた後
    滑らかになるよう補間処理を行います
    """
    x_size, x_dim = xs.shape[0], xs.shape[1]
    if random_seed is not None:
        np.random.seed(random_seed)
    x = np.tile(np.arange(0, x_size, (x_size - 1) / (n_knots + 1)).reshape(-1, 1), reps=(1, x_dim))
    # print("generate_random_curves:sigma", sigma)
    y = np.random.normal(loc=1.0, scale=sigma, size=(n_knots + 2, x_dim))
    x_folds = []
    for i in range(x_dim):
        func = interp1d(x[:, i], y[:, i], kind='linear')
        x_folds.append(func(np.arange(x_size)))
    return np.array(x_folds).transpose()


def magnitude_warp(xs, sigma=0.2, individual=False, random_seed=None):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param sigma:
    :param individual: bool 個別にスケーリングするか否か
    :param random_seed:
    :return: numpy.array [-1, n_dim]
    データ系列毎にデータ全体を１を中心にn_knotsだけ折り曲げた後
    滑らかになるよう補間処理をしたものとデータを乗じます
    """
    x_size, x_dim = xs.shape[0], xs.shape[1]
    if random_seed is not None:
        np.random.seed(random_seed)
    if individual:
        return xs * generate_random_curves(xs, sigma=sigma)
    else:
        return xs * generate_random_curves(np.zeros((x_size, 1)), sigma=sigma)


class TsMagnitudeWarp(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=0.2, individual=False, random_seed=None):
        self.sigma = sigma
        self.individual = individual
        self.random_curve = None
        self.random_seed = random_seed

    def fit(self, xs):
        x_size, x_dim = xs.shape[0], xs.shape[1]
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if self.individual:
            self.random_curve = generate_random_curves(xs, self.sigma)
        else:
            self.random_curve = generate_random_curves(np.zeros((x_size, 1)), self.sigma)
        return self

    def transform(self, xs):
        return xs * self.random_curve


def distort_time_steps(xs, sigma=0.2, n_knots=4, random_seed=None):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param sigma:
    :param n_knots: 折り曲げの回数
    :param random_seed:
    :return: numpy.array [-1, n_dim]
    データ系列毎にデータ発生時刻をにn_knotsだけ折り曲げた後(非等間隔)
    発生時刻が滑らかになるように補間処理して等間隔の発生時刻を取得する
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # TODO
    tt = generate_random_curves(xs, sigma=0.2, n_knots=n_knots)  # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = (xs.shape[0] - 1) / tt_cum[-1, :]
    tt_cum = tt_cum * t_scale
    return tt_cum


def time_warp(xs, sigma=0.2, n_knots=4, individual=False, kind="linear", random_seed=None):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param sigma:
    :param n_knots:
    :param individual: bool 個別にスケーリングするか否か
    :param kind:
    :param random_seed:
    :return: numpy.array [-1, n_dim]
    データ系列毎にデータ発生時刻を滑らかにシフトさせる
    先頭と末尾のデータの発生時刻は固定
    """
    x_size, x_dim = xs.shape[0], xs.shape[1]
    if random_seed is not None:
        np.random.seed(random_seed)
    xs_new = []  # return array
    if individual:
        for i in range(x_dim):
            print("time_warp:individual:sigma", sigma)
            tt_new = distort_time_steps(xs, sigma=sigma, n_knots=n_knots)  # 非等間隔の時系列
            func = interp1d(tt_new[:, i], xs[:, i], kind=kind, fill_value="extrapolate")
            xs_new.append(func(np.arange(x_size)))  # 等間隔になるようにデータ補間
        return np.array(xs_new).transpose()
    else:
        xs_0 = xs[:, 0].reshape(-1, 1)
        tt_new = distort_time_steps(xs_0, sigma=sigma, n_knots=n_knots)  # 非等間隔の時系列
        for i in range(x_dim):
            func = interp1d(tt_new[:, 0], xs[:, i], kind=kind, fill_value="extrapolate")
            xs_new.append(func(np.arange(x_size)))  # 等間隔になるようにデータ補間
        return np.array(xs_new).transpose()


class TsTimeWarp(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=0.2, n_knots=4, individual=False, kind="linear", random_seed=None):
        self.sigma = sigma
        self.n_knots = n_knots
        self.individual = individual
        self.kind = kind
        self.funcs = None
        self.random_seed = random_seed

    def fit(self, xs):
        x_size, x_dim = xs.shape[0], xs.shape[1]
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.funcs = []
        if self.individual:
            for i in range(x_dim):
                tt_new = distort_time_steps(xs, self.sigma, self.n_knots)  # 非等間隔の時系列
                func = interp1d(tt_new[:, i], xs[:, i], kind=self.kind, fill_value="extrapolate")
                self.funcs.append(func)
        else:
            xs_0 = xs[:, 0].reshape(-1, 1)
            tt_new = distort_time_steps(xs_0, self.sigma, self.n_knots)  # 非等間隔の時系列
            for i in range(x_dim):
                func = interp1d(tt_new[:, 0], xs[:, i], kind=self.kind, fill_value="extrapolate")
                self.funcs.append(func)
        return self

    def transform(self, xs):
        x_size, x_dim = xs.shape[0], xs.shape[1]
        xs_new = []
        for i, func in enumerate(self.funcs):
            xs_new.append(func(np.arange(x_size)))
        return np.array(xs_new).transpose()


def time_warp_Xy(xs, ys, sigma=0.2, n_knots=4, individual=False, kind="linear"):
    """
    :return:
    :param xs: Data:numpy.array [-1, n_dim]
    :param ys: Data:numpy.array [-1, 1]
    :param sigma:
    :param n_knots:
    :param individual: bool 個別にスケーリングするか否か
    :param kind:
    :return: numpy.array [-1, n_dim]
    データ系列毎にデータ発生時刻を滑らかにシフトさせる
    先頭と末尾のデータの発生時刻は固定
    """
    x_size, x_dim = xs.shape[0], xs.shape[1]
    if individual:
        tt_new = distort_time_steps(xs, sigma=sigma, n_knots=n_knots)  # 非等間隔の時系列
        xs_new = []
        for i in range(x_dim):
            func = interp1d(tt_new[:, i], xs[:, i], kind=kind, fill_value="extrapolate")
            xs_new.append(func(np.arange(x_size)))  # 等間隔になるようにデータ補間

        tt_new = distort_time_steps(ys, sigma=sigma, n_knots=n_knots)
        func = interp1d(tt_new[:, 0], ys[:, 0], kind="nearest", fill_value="extrapolate")
        ys_new = func(np.arange(x_size))
        return np.array(xs_new).transpose(), ys_new
    else:
        xs_std = xs[:, 0].reshape(-1, 1)
        tt_new = distort_time_steps(xs_std, sigma=sigma, n_knots=n_knots)  # 非等間隔の時系列
        xs_new = []
        for i in range(x_dim):
            func = interp1d(tt_new[:, 0], xs[:, i], kind=kind, fill_value="extrapolate")
            xs_new.append(func(np.arange(x_size)))  # 等間隔になるようにデータ補間

        func = interp1d(tt_new[:, 0], ys[:, 0], kind="nearest", fill_value="extrapolate")
        ys_new = func(np.arange(x_size))
        return np.array(xs_new).transpose(), ys_new.reshape(-1, 1)


def time_stretch(xs, scale=2., kind='linear', debug_flag=False):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param scale: flot
    :param kind: string
    :param debug_flag:
    :return:
    時系列方向にscale倍に伸縮した配列を返す、時系列方向の値の数はscale倍される
    """
    assert scale > 0

    x_size, x_dim = xs.shape[0], xs.shape[1]
    xs_new = []

    for i in range(x_dim):
        ts_new = np.arange(0, x_size * scale, scale)
        if debug_flag:
            print(ts_new.shape, xs[:, i].shape)
        func = interp1d(ts_new, xs[:, i], kind=kind, fill_value="extrapolate")
        xs_new.append(func(np.arange(int(x_size * scale))))  # 等間隔になるようにデータ補間
    return np.array(xs_new).transpose()


class TsTimeStretch(BaseEstimator, TransformerMixin):
    def __init__(self, scale=2., kind="linear", debug_flag=False):
        assert scale > 0
        self.scale = scale
        self.kind = kind
        self.debug_flag = debug_flag
        self.funcs = None

    def fit(self, xs):
        x_size, x_dim = xs.shape[0], xs.shape[1]
        self.funcs = []
        for i in range(x_dim):
            ts_new = np.arange(0, x_size * self.scale, self.scale)
            if self.debug_flag:
                print(ts_new.shape, xs[:, i].shape)
            func = interp1d(ts_new, xs[:, i], kind=self.kind, fill_value="extrapolate")
            self.funcs.append(func)
        return self

    def transform(self, xs):
        x_size, x_dim = xs.shape[0], xs.shape[1]
        xs_new = []
        for func in self.funcs:
            xs_new.append(func(np.arange(int(x_size * self.scale))))
        return np.array(xs_new).transpose()


def partial_time_stretch(xs, stretches=((0., 1., 1.),), kind='linear', debug_flag=False):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param stretches: ((start, end, scale),...)
    :param kind:
    :param debug_flag:
    :return:
    時系列方向にscale倍に伸縮した配列を返す、時系列方向の値の数はscale倍される
    """
    x_size, x_dim = xs.shape[0], xs.shape[1]
    xs_new = []
    ts_diff = np.ones(x_size)
    ts_new = None
    for (start, end, stretch) in stretches:
        ts_diff[int(start * x_size):int(end * x_size)] = stretch
        ts_new = ts_diff.cumsum()

    for i in range(x_dim):
        if debug_flag:
            print(ts_diff.shape, xs[:, i].shape)
        func = interp1d(ts_new, xs[:, i], kind=kind, fill_value="extrapolate")
        xs_new.append(func(np.arange(int(ts_new[-1]))))  # 等間隔になるようにデータ補間
    return np.array(xs_new).transpose()


class TsPartialTimeStretch(BaseEstimator, TransformerMixin):
    def __init__(self, stretches, kind="linear", debug_flag=False):
        self.stretches = stretches
        self.kind = kind
        self.debug_flag = debug_flag
        self.ts_new = None
        self.funcs = None

    def fit(self, xs):
        self.funcs = []
        x_size, x_dim = xs.shape[0], xs.shape[1]
        ts_diff = np.ones(x_size)
        for (start, end, stretch) in self.stretches:
            ts_diff[int(start * x_size):int(end * x_size)] = stretch
        self.ts_new = ts_diff.cumsum()
        for i in range(x_dim):
            if self.debug_flag:
                print(self.ts_new.shape, xs[:, i].shape)
            func = interp1d(self.ts_new, xs[:, i], kind=self.kind, fill_value="extrapolate")
            self.funcs.append(func)
        return self

    def transform(self, xs):
        xs_new = []
        for func in self.funcs:
            xs_new.append(func(np.arange(int(self.ts_new[-1]))))
        return np.array(xs_new).transpose()


def resampling(xs, src_period, dst_period, kind='linear'):
    """
    :param xs: Data:numpy.array [-1, n_dim]
    :param src_period: flot 元の時系列データのサンプリング周期
    :param dst_period: flot 目的のの時系列データのサンプリング周期
    :param kind: string
    :return:
    時系列データを目的のサンプリング周期でリサンプリングする
    """
    x_size, x_dim = xs.shape[0], xs.shape[1]
    xs_new = []

    for i in range(x_dim):
        func = interp1d(np.arange(x_size), xs[:, i], kind=kind, fill_value="extrapolate")
        xs_new.append(func(np.arange(0, x_size, dst_period / src_period)))  # 等間隔になるようにデータ補間
    return np.array(xs_new).transpose()


class TsResampling(BaseEstimator, TransformerMixin):
    def __init__(self, src_period, dst_period, kind="linear"):
        self.src_period = src_period
        self.dst_period = dst_period
        self.kind = kind

    def fit(self, xs):
        return self

    def transform(self, xs):
        x_size, x_dim = xs.shape[0], xs.shape[1]
        xs_new = []
        for i in range(x_dim):
            func = interp1d(np.arange(x_size), xs[:, i], kind=self.kind, fill_value="extrapolate")
            xs_new.append(func(np.arange(0, x_size, self.dst_period / self.src_period)))  # 等間隔になるようにデータ補間
        return np.array(xs_new).transpose()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N = 1000
    X = np.arange(N)
    Y1 = np.sin(2 * np.pi * X / N * 2).reshape(-1, 1)
    Y2 = np.cos(2 * np.pi * X / N * 2).reshape(-1, 1)
    Y = np.concatenate([Y1, Y2], axis=1)
    # Y1 = magnitude_warp(Y, individual=False)
    #
    # sc = TsScalar(sigma=0.5)
    # Y2 = sc.fit_transform(Y1)
    #
    # st = TsTimeStretch(scale=0.8)
    # Y3 = st.fit_transform(Y1)

    # plt.plot(X, Y1)
    # plt.plot(X, Y2)
    # plt.plot(range(len(Y3)), Y3)
    # plt.show()
    # print("hello")

    # y_ans = time_stretch(Y, scale=0.5)
    trans = TsTimeStretch(scale=0.8)

    trans.fit(Y1)
    y_ans = trans.transform(Y2)
    print(Y.shape)
    print(y_ans.shape)

    # np.testing.assert_array_equal(y_ans, trans.fit_transform(Y))

    # stretches = [(0.2, 0.5, 1.2)]  # ,(0.5,0.7,0.2)]
    # X_ans = partial_time_stretch(Y, stretches, debug_flag=True)
    # plt.plot(y_ans)
    # plt.plot(Y1)
    plt.plot(Y1)
    # plt.plot(Y1)
    # plt.plot(Y2)
    plt.plot(y_ans)
    plt.show()
