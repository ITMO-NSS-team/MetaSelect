from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ms.utils.typing import NDArrayFloatT
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


@dataclass
class PILOTResult:
    x_bar: NDArrayFloatT
    x_hat: NDArrayFloatT
    x_0: NDArrayFloatT
    alpha: NDArrayFloatT
    optim: NDArrayFloatT
    perf: NDArrayFloatT
    a: NDArrayFloatT
    b: NDArrayFloatT
    c: NDArrayFloatT
    z: NDArrayFloatT
    error: NDArrayFloatT
    r2: NDArrayFloatT
    summary: pd.DataFrame


class InstanceSpaceAnalysis:
    def __init__(
            self,
            n_tries: int,
    ):
        self.n_tries = n_tries


    def pilot(
            self,
            features: pd.DataFrame,
            metrics: pd.DataFrame,
    ) -> PILOTResult:
        print("Started PILOT")
        x = features.to_numpy(copy=True)
        y = metrics.to_numpy(copy=True)
        n = x.shape[1]
        x_bar = np.hstack((x, y))
        m = x_bar.shape[1]
        p_dist = pdist(x).reshape(-1, 1)

        x_0 = 2 * np.random.rand(2 * m + 2 * n, self.n_tries) - 1

        alpha = np.zeros((2 * m + 2 * n, self.n_tries))
        optim = np.zeros((1, self.n_tries))
        perf = np.zeros((1, self.n_tries))

        print("Performing BFGS optimization")

        for i in range(self.n_tries):
            result = minimize(
                self.error_func,
                x_0[:, i],
                args=(x_bar, n, m),
                method='BFGS',
                options={'disp': False}
            )
            alpha[:, i] = result.x
            optim[:, i] = result.fun
            aux = alpha[:, i]
            a = np.reshape(aux[:2 * n], (2, n))
            z = np.dot(x, a.T)
            perf[:, i] = np.corrcoef(p_dist.flatten(), pdist(z).flatten())[0, 1]

        print("Computing z1, z2")

        idx = np.argmax(perf)

        a = np.reshape(alpha[:2 * n, idx], (2, n))
        z = np.dot(x, a.T)
        b = np.reshape(alpha[2 * n:, idx], (m, 2))
        x_hat = np.dot(z, b.T)
        c = b[n:m, :].T
        b = b[:n, :]
        error = np.sum((x_bar - x_hat) ** 2)
        r2 = np.diag(np.corrcoef(x_bar.T, x_hat.T)[0, 1:]) ** 2

        summary = pd.DataFrame(
            data=np.round(a, 4).T,
            columns=["z1", "z2"],
            index=features.columns
        )

        return PILOTResult(
            x_bar=x_bar,
            x_hat=x_hat,
            x_0=x_0,
            alpha=alpha,
            optim=optim,
            perf=perf,
            a=a,
            b=b,
            c=c,
            z=z,
            error=error,
            r2=r2,
            summary=summary,
        )

    @staticmethod
    def error_func(
            alpha: NDArrayFloatT,
            x_bar: NDArrayFloatT,
            n: int,
            m: int,
    ):
        a = np.reshape(alpha[:2 * n], (2, n))
        b = np.reshape(alpha[2 * n:], (m, 2))
        residual = x_bar - (b @ a @ x_bar[:, :n].T).T
        return np.nanmean(np.nanmean(residual ** 2, axis=1), axis=0)
