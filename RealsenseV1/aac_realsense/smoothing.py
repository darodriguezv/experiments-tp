"""Smoothers, all of them confidence-aware.

The webcam pipeline had no notion of a bad frame: the regression always returned
a point, so every sample entered the filter with equal weight and a mislocated
iris pulled the cursor as hard as a good reading. The geometric model produces a
confidence with nameable causes - depth lost, poor rigid fit, iris too small,
gaze too eccentric - and a filter that ignores that is discarding the most
useful thing the new pipeline provides.

So confidence scales the measurement noise. A low-confidence frame nudges the
estimate instead of yanking it, which is exactly the behaviour a dwell-selected
board wants: it keeps the cursor parked on the tile the user is holding rather
than flicking away and resetting the dwell.

Smoothing hides jitter, it does not remove it. Reach for a stronger smoother
only after `accuracy` says the problem is jitter rather than offset, since every
setting here trades latency for steadiness and latency is felt directly by
someone typing a sentence one dwell at a time.
"""

from __future__ import annotations

import numpy as np


class Smoother:
    """Interface: feed a measurement, get a filtered position."""

    def step(self, x: float, y: float, confidence: float = 1.0
             ) -> tuple[float, float]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class NoSmoother(Smoother):
    """Pass-through. Use it to see the true jitter before hiding any of it."""

    def step(self, x, y, confidence: float = 1.0):
        return float(x), float(y)


class EMASmoother(Smoother):
    """Exponential moving average, with the rate scaled by confidence.

    `alpha` is the weight of the *new* measurement, so larger is more
    responsive. A low-confidence frame gets its alpha scaled down rather than
    being dropped: dropping frames stalls the cursor, and a stalled cursor
    during a dwell reads to the user as the system having frozen.
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._state: np.ndarray | None = None

    def step(self, x, y, confidence: float = 1.0):
        z = np.array([float(x), float(y)])
        if self._state is None:
            self._state = z
            return float(z[0]), float(z[1])
        a = self.alpha * float(np.clip(confidence, 0.05, 1.0))
        self._state = (1.0 - a) * self._state + a * z
        return float(self._state[0]), float(self._state[1])

    def reset(self) -> None:
        self._state = None


class KalmanSmoother(Smoother):
    """Constant-velocity Kalman filter over the screen point.

    Returns the **corrected** estimate, not the predicted one. That distinction
    matters: a filter reporting its prediction leads the true gaze, which looks
    pleasingly responsive on a moving cursor and is wrong for dwell, because it
    crosses a tile boundary before the eye does and can fire the neighbouring
    tile. On an AAC board a mis-selection costs a delete and a retry, so the
    conservative choice is the correct one.

    A velocity model rather than a position-only one because gaze moves in
    saccades: a position-only filter treats every saccade as noise and lags
    badly through it.
    """

    def __init__(self, process_var: float = 900.0, measure_var: float = 400.0,
                 dt: float = 1.0 / 30.0):
        self.dt = float(dt)
        self.process_var = float(process_var)
        self.measure_var = float(measure_var)
        self._x: np.ndarray | None = None
        self._P = np.eye(4) * 1e4

        self._F = np.array([[1.0, 0.0, self.dt, 0.0],
                            [0.0, 1.0, 0.0, self.dt],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
        self._H = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]])

        # Continuous white-noise-acceleration discretisation. Using this rather
        # than a diagonal guess is what keeps position and velocity uncertainty
        # correlated, which is where a constant-velocity filter gets its ability
        # to ride through a saccade instead of lagging behind it.
        dt2, dt3, dt4 = self.dt ** 2, self.dt ** 3, self.dt ** 4
        block = np.array([[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]])
        self._Q = np.zeros((4, 4))
        for i in (0, 1):
            self._Q[np.ix_([i, i + 2], [i, i + 2])] = block * self.process_var

    def step(self, x, y, confidence: float = 1.0):
        z = np.array([float(x), float(y)])
        if self._x is None:
            self._x = np.array([z[0], z[1], 0.0, 0.0])
            self._P = np.eye(4) * 1e3
            return float(z[0]), float(z[1])

        # Predict.
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        # Update, with the measurement noise inflated as confidence drops. The
        # quadratic makes a half-confidence frame count for a quarter as much,
        # which matches how quickly gaze error grows once the iris gets small.
        conf = float(np.clip(confidence, 0.05, 1.0))
        R = np.eye(2) * (self.measure_var / (conf * conf))

        y_res = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y_res
        self._P = (np.eye(4) - K @ self._H) @ self._P
        return float(self._x[0]), float(self._x[1])

    def reset(self) -> None:
        self._x = None
        self._P = np.eye(4) * 1e4


class KalmanEMASmoother(Smoother):
    """Kalman followed by an EMA. The default.

    The Kalman handles the dynamics; the EMA takes off the residual
    frame-to-frame shimmer that a velocity model leaves behind. Set
    `ema_alpha=1.0` to disable the second stage.
    """

    def __init__(self, ema_alpha: float = 0.25, **kalman_kwargs):
        self.kalman = KalmanSmoother(**kalman_kwargs)
        # An EMA weight near 0 in the usual "alpha is the new sample" reading
        # means "almost no new information", so the flag is inverted here to
        # match how it reads in the CLI, where higher is smoother.
        self.ema = EMASmoother(alpha=1.0 - float(np.clip(ema_alpha, 0.0, 0.95)))

    def step(self, x, y, confidence: float = 1.0):
        kx, ky = self.kalman.step(x, y, confidence)
        return self.ema.step(kx, ky, confidence)

    def reset(self) -> None:
        self.kalman.reset()
        self.ema.reset()


class MedianSmoother(Smoother):
    """Median of a short window. The most stable option for dwell.

    A median is immune to a single bad frame in a way no linear filter is: one
    wild sample cannot move it at all, whereas it shifts an EMA by alpha times
    the error. The cost is latency of about half the window, and a median lags
    through a saccade rather than following it - which is a reasonable trade for
    a board with large tiles and a poor one for a moving cursor.

    Confidence is used to weight frames by repetition, which is the only way a
    median can be weighted at all.
    """

    def __init__(self, window: int = 7):
        self.window = int(max(1, window))
        self._xs: list[float] = []
        self._ys: list[float] = []

    def step(self, x, y, confidence: float = 1.0):
        # A frame counts twice when it is trustworthy, once otherwise. Crude,
        # but it keeps the estimator a genuine median rather than something
        # that merely resembles one.
        repeats = 2 if confidence >= 0.6 else 1
        for _ in range(repeats):
            self._xs.append(float(x))
            self._ys.append(float(y))
        limit = self.window * 2
        del self._xs[:-limit]
        del self._ys[:-limit]
        return float(np.median(self._xs)), float(np.median(self._ys))

    def reset(self) -> None:
        self._xs.clear()
        self._ys.clear()


def make(name: str, ema_alpha: float = 0.25, fps: float = 30.0) -> Smoother:
    """Build a smoother by name, matching the CLI's `--smoother` values."""
    dt = 1.0 / max(fps, 1.0)
    if name == "none":
        return NoSmoother()
    if name == "ema":
        return EMASmoother(alpha=1.0 - float(np.clip(ema_alpha, 0.0, 0.95)))
    if name == "kalman":
        return KalmanSmoother(dt=dt)
    if name == "kalman_ema":
        return KalmanEMASmoother(ema_alpha=ema_alpha, dt=dt)
    if name == "median":
        return MedianSmoother()
    raise ValueError(f"suavizado desconocido {name!r}")
