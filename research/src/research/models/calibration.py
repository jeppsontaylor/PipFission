"""Probability calibration. Sigmoid (Platt) is preferred on the
1000-bar window because isotonic overfits when calibration data is
sparse — this matches the guidance from López de Prado / scikit-learn.
"""
from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV


def wrap_sigmoid(estimator: Any, cv: int = 5) -> CalibratedClassifierCV:
    """Wrap a fitted-or-unfitted estimator in `CalibratedClassifierCV`
    using sigmoid calibration. Caller passes a CV integer; the
    research training driver typically uses an internal purged k-fold
    split for the calibrator's CV (see training/side_train.py).
    """
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=cv)
