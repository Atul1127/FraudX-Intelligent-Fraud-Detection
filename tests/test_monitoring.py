import numpy as np

from src.monitoring.drift import drift_level, prediction_shift, psi


def test_psi_is_near_zero_for_same_distribution():
    reference = np.linspace(0.05, 0.95, 100)
    assert psi(reference, reference) < 1e-9


def test_drift_bands():
    assert drift_level(0.05) == "stable"
    assert drift_level(0.15) == "warning"
    assert drift_level(0.30) == "drift"


def test_prediction_shift_detects_distribution_change():
    reference = np.linspace(0.01, 0.20, 100)
    current = np.linspace(0.70, 0.99, 100)
    result = prediction_shift(reference, current)
    assert result["psi"] > 0.25
    assert result["level"] == "drift"
