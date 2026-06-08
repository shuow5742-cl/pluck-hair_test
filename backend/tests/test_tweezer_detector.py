from __future__ import annotations

import math

import numpy as np

from src.core.tweezer_detector import (
    TweezerConfig,
    TweezerDetector,
    _BranchSupportModel,
)


def test_predict_closed_tip_from_pair_uses_asymmetric_open_ratio():
    detector = TweezerDetector(
        TweezerConfig(
            open_pick_ratio=0.32,
            open_forward_ratio=0.02,
            open_forward_max_px=8.0,
        )
    )
    axis = np.array([-1.0, 0.0], dtype=float)
    upper = (505.575, 378.225)
    lower = (522.9787234042553, 443.63829787234044)

    predicted = detector._predict_closed_tip_from_pair(
        upper,
        lower,
        axis,
        is_open=True,
    )

    # Same-pose closed/open calibration pair supplied with the task.
    np.testing.assert_allclose(
        [predicted[0], predicted[1]],
        [509.79, 399.16],
        atol=0.75,
    )


def test_stabilize_refined_tips_limits_defocus_lower_branch_drift():
    detector = TweezerDetector(
        TweezerConfig(
            open_tip_max_backtrack_px=18.0,
            open_tip_max_lateral_drift_px=18.0,
        )
    )
    axis = np.array([-1.0, 0.0], dtype=float)
    model = _BranchSupportModel(
        upper_m=-0.19715379895679272,
        upper_b=-1205.5311977903916,
        lower_m=0.4894070603428305,
        lower_b=-215.07467170833507,
        eval_along=-1579.0,
        support_upper_tip=(1579.0, 894.2253492376159),
        support_lower_tip=(1579.0, 987.8484199896643),
    )
    refined = (
        (1575.3090909090909, 913.4636363636364),
        (1601.1868131868132, 1009.8681318681319),
    )

    stabilized = detector._stabilize_refined_tips(refined, model, axis)
    lower = stabilized[1]

    assert lower[0] < refined[1][0]
    assert lower[1] < refined[1][1]
    assert math.dist(lower, model.support_lower_tip) < math.dist(
        refined[1], model.support_lower_tip
    )
