import numpy as np
from scipy.optimize import minimize


def PILOT(
        x: np.ndarray,
        y: np.ndarray,
        features_names: list[str],
        num_tries: int = 10,
):
    num_features = x.shape[1]
    x_bar = np.hstack([x, y])
    num_all = x_bar.shape[1]

    # np.random.seed(0)
    x_0 = 2 * np.random.rand(2 * num_all + 2 * num_features, num_tries) - 1

    results = []
    for i in range(x_0.shape[1]):
        res = minimize(
            error_func,
            x_0[:, i],
            args=(x_bar, num_features, num_all),
            method='L-BFGS-B',
            options={'disp': False}
        )
        results.append((res.x, res.fun))

    best_idx = np.argmin([res[1] for res in results])
    alpha = results[best_idx][0]

    out = {'A': alpha[:2 * num_features].reshape(2, num_features)}
    out['Z'] = x @ out['A'].T
    b_c = alpha[2 * num_features:].reshape(num_all, 2)
    out['B'] = b_c[:num_features, :]
    out['C'] = b_c[num_features:, :].T
    x_hat = out['Z'] @ b_c.T
    out['error'] = np.sum((x_bar - x_hat) ** 2)
    out['R2'] = np.diag(np.corrcoef(x_bar.T, x_hat.T)[:num_features, num_features:]) ** 2

    out['summary'] = np.zeros((3, num_features + 1), dtype=object)
    out['summary'][0, 1:] = features_names
    out['summary'][1:, 0] = ['Z_{1}', 'Z_{2}']
    out['summary'][1:, 1:] = np.round(out['A'], 4)
    print(out['summary'])

    return out


def error_func(alpha: float, x_bar: np.ndarray, n: int, m: int) -> float:
    a = alpha[ : 2 * n].reshape(2, n)
    b_c = alpha[2 * n : ].reshape(m, 2)
    z = a @ x_bar[:, : n].T
    x_hat = (b_c @ z).T
    return np.nanmean((x_bar - x_hat) ** 2)
