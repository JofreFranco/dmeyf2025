import os
import logging
from typing import Any, Optional
import gc
import pandas as pd
import numpy as np
import math
from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

logger = logging.getLogger(__name__)

class SamplerProcessor:
    """
    Realiza sampling mensual basado en una curva precalculada
    (exponencial o uniforme)
    """

    def __init__(
        self,
        method="uniforme",
        p0=1.0,
        target_sr=0.2,
        special_months=None,  # dict: {mes: sampling_rate}
        month_col="foto_mes",
        verbose=False,
    ):
        self.method = method.lower()
        self.p0 = float(p0)
        self.target_sr = float(target_sr)
        self.special_months = special_months or {}
        self.month_col = month_col
        self.verbose = verbose
        self.curve_ = None

        self._initialize_curve()

    def _initialize_curve(self):
        if self.method == "uniform":
            curve = self._compute_uniform_curve()

        elif self.method == "exponential":
            curve = self._compute_exponential_curve()

        else:
            raise ValueError("Método de sampleo no soportado: %s" % self.method)

        curve = self._apply_special_months(curve)

        self.curve_ = curve

        if self.verbose:
            print("Curva de sampling final:")
            print(self.curve_)

    def _compute_uniform_curve(self):
        """
        Curva constante: todo a target_sr.
        """
        return lambda months: np.full_like(months, fill_value=self.target_sr, dtype=float)

    def _compute_exponential_curve(self):
        """
        Crea una curva exponencial p(t) = p0 * exp(-k t)
        donde k se ajusta para que el promedio sea target_sr.
        """

        def curve_fn(months):
            # Normalizamos meses -> t = distancia desde max
            t = months.max() - months

            # Resolver k tal que promedio == target_sr
            # avg = (1/(T+1)) * sum( p0 * exp(-k t) )
            # => encontrar k por bisección

            T = t.max()

            def avg_for_k(k):
                return np.mean(self.p0 * np.exp(-k * t))

            lo, hi = 0.0, 10.0  # ojo si el valor del decay termina muy cerca de 10

            for _ in range(60):  # bisección
                mid = (lo + hi) / 2
                if avg_for_k(mid) > self.target_sr:
                    lo = mid
                else:
                    hi = mid

            k_star = (lo + hi) / 2
            return self.p0 * np.exp(-k_star * t)

        return curve_fn

    def _apply_special_months(self, curve_fn):
        """
        Devuelve un nuevo callable que incorpora override
        de sampling rate para meses específicos.
        """

        def new_curve(months):
            base = curve_fn(months).copy()

            for m, p in self.special_months.items():
                base[months == m] = p

            return base

        return new_curve

    def transform(self, df: pd.DataFrame):
        """
        Aplica sampling SOLO a las filas con label = 0.
        Las filas con label = 1 se conservan siempre.
        """
        df = df.copy()
        
        df_pos = df[df["label"] == 1]
        df_neg = df[df["label"] == 0]
        
        months_neg = df_neg[self.month_col].values
        probs = self.curve_(months_neg)
        
        keep_neg = np.random.rand(len(df_neg)) < probs
        df_neg_sampled = df_neg[keep_neg]
        out = pd.concat([df_pos, df_neg_sampled], ignore_index=True)
        
        return out
