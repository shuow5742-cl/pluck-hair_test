"""Unit tests for verify_transform helpers."""

from __future__ import annotations

import math

from src.core.coordinate_transform import CoordinateTransformer, ExtrinsicCalibration
from tools.calibrate.verify_transform import build_report, measure_point, summarize_samples


def _make_cal() -> ExtrinsicCalibration:
    return ExtrinsicCalibration(
        mm_per_pixel=0.01,
        dx=1.0,
        dy=-2.0,
        cx=500.0,
        cy=500.0,
        flip_y=False,
    )


def test_measure_point_calculates_world_and_error():
    cal = _make_cal()
    transformer = CoordinateTransformer(cal)

    sample = measure_point(
        transformer=transformer,
        pixel=(600.0, 450.0),
        arm_pose=(10.0, 20.0),
        expected_world=(12.0, 17.0),
        sample_id=1,
    )

    # x = 10 + 1 + (600 - 500) * 0.01 = 12
    # y = 20 - 2 + (450 - 500) * 0.01 = 17.5
    assert math.isclose(sample.measured_x, 12.0, abs_tol=1e-9)
    assert math.isclose(sample.measured_y, 17.5, abs_tol=1e-9)
    assert math.isclose(sample.error_x, 0.0, abs_tol=1e-9)
    assert math.isclose(sample.error_y, 0.5, abs_tol=1e-9)
    assert math.isclose(sample.error_norm, 0.5, abs_tol=1e-9)


def test_summarize_samples_aggregates_error_metrics():
    cal = _make_cal()
    transformer = CoordinateTransformer(cal)
    samples = [
        measure_point(
            transformer=transformer,
            pixel=(500.0, 500.0),
            arm_pose=(0.0, 0.0),
            expected_world=(1.0, -2.0),
            sample_id=1,
        ),
        measure_point(
            transformer=transformer,
            pixel=(510.0, 490.0),
            arm_pose=(0.0, 0.0),
            expected_world=(1.2, -2.2),
            sample_id=2,
        ),
    ]

    summary = summarize_samples(samples)
    assert summary["num_samples"] == 2
    assert math.isclose(summary["mean_error_x_mm"], -0.05, abs_tol=1e-9)
    assert math.isclose(summary["mean_error_y_mm"], 0.05, abs_tol=1e-9)
    assert math.isclose(summary["mean_abs_error_x_mm"], 0.05, abs_tol=1e-9)
    assert math.isclose(summary["mean_abs_error_y_mm"], 0.05, abs_tol=1e-9)
    assert math.isclose(summary["max_error_norm_mm"], 0.1414, abs_tol=1e-9)


def test_build_report_contains_calibration_and_samples():
    cal = _make_cal()
    transformer = CoordinateTransformer(cal)
    sample = measure_point(
        transformer=transformer,
        pixel=(500.0, 500.0),
        arm_pose=(0.0, 0.0),
        expected_world=(1.0, -2.0),
        sample_id=1,
    )

    report = build_report(
        samples=[sample],
        calibration=cal,
        extrinsic_path="config/calibration/extrinsic.yaml",
        intrinsic_path="config/calibration/camera_intrinsic.yaml",
        config_path="config/settings.dev.yaml",
        image_path=None,
        preview_scale=0.5,
    )

    assert report["mode"] == "camera"
    assert report["summary"]["num_samples"] == 1
    assert report["calibration"]["mm_per_pixel"] == 0.01
    assert report["samples"][0]["sample_id"] == 1
    assert report["samples"][0]["error_norm"] == 0.0
