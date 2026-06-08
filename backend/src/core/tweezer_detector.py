"""Classical-CV tweezer tip detector (镊子尖部识别).

This module is *new* in pluck-hair_test. It is deliberately model-free: we do
not yet have labelled tweezer data, so the tip is located with thresholding +
contour / convexity-defect analysis instead of a neural keypoint model. Every
threshold lives in :class:`TweezerConfig` so the operator can field-tune it from
``config/settings.test.yaml`` without touching code — the same philosophy the
production pipeline uses for ``crop_single_square`` / ``abstain_near_metal``.

Two physical states must be handled (see the reference frames in the design
discussion):

* **Closed** — the two blades have converged to a single sharp point. The tip is
  the single deepest contour vertex along the "into the frame" axis.
* **Open** — the two blades form a ``V``. We find the deepest convexity defect in
  the *leading* portion of the contour (the notch between the two prongs) and
  take the two prong tips as the defect's start / end hull vertices. The reported
  pick point is **not** the midpoint of the two tips but an offset point on the
  connecting line (``open_pick_ratio``), matching where the bench crosshair sits
  in the open-tweezer reference image.

The detector returns a :class:`TweezerResult` in *original full-frame* pixel
coordinates — the same coordinate system the segmentation pick points live in —
so a pixel distance between the two can be scaled to millimetres with a single
``mm_per_pixel`` factor (the rig uses a telecentric lens, so magnification is
depth-invariant and the surface-plane scale is valid at tip height too).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


@dataclass
class TweezerConfig:
    """Tunable parameters for :class:`TweezerDetector`.

    All distances are in *pixels of the original camera frame* unless the name
    ends in ``_mm`` or ``_ratio`` / ``_frac``.
    """

    enabled: bool = True

    # ---- segmentation of the metal body ---------------------------------------
    # On this rig the steel tweezer body reads DARKER (~110-130) than the bright
    # back-plate (~160-226), so "dark" (default) — a gray < T cut — isolates it
    # cleanly. "edge" thresholds gradient magnitude (smooth plate ≈ 0, textured
    # metal high) then closes the outline; "bright" is the classic gray >= T cut.
    # Pick whichever matches the lighting and field-tune the threshold.
    # "dark_adaptive" (default) thresholds at (background − dark_margin), where
    # background is a high percentile of the gray image. This tracks the tweezer
    # across focus: a sharply-focused blade is clearly dark, but a SEVERELY
    # DEFOCUSED blade is only slightly darker than the plate — a fixed cut misses
    # it, while "background minus a margin" still catches it. "dark" is the old
    # fixed cut; "edge"/"bright" are alternatives.
    seg_method: str = "dark_adaptive"    # dark_adaptive | dark | edge | bright
    bg_percentile: float = 80.0          # background (plate) gray level percentile
    dark_margin: int = 26                # tweezer = gray < (background − this)
    dark_threshold: int = 128            # gray < this is metal (seg_method=dark)
    bright_threshold: int = 170          # gray >= this is metal (seg_method=bright)
    use_otsu: bool = False               # auto-threshold instead (seg_method=bright)
    edge_threshold: int = 40             # gradient-magnitude cut (seg_method=edge)
    edge_close_ksize: int = 25           # close to fill the metal body outline
    blur_ksize: int = 5                  # odd; <=1 disables the pre-blur
    morph_open_ksize: int = 5            # remove debris/speckle so the tweezer is isolated
    morph_close_ksize: int = 15          # bridge blade gaps (<=1 disables)

    # ---- candidate blob filtering --------------------------------------------
    min_area_px: int = 4000              # reject reflections / small foreign bits
    max_area_ratio: float = 0.60         # reject if blob covers >60% of frame
    require_border_touch: bool = True    # the tweezer enters from a frame edge
    border_band_px: int = 10             # "touching" tolerance

    # ---- entry side / tip direction ------------------------------------------
    # Which edge the tweezer enters from. "auto" infers it from where the blob
    # mass concentrates against the border each frame.
    entry_side: str = "auto"             # auto | left | right | top | bottom

    # ---- open vs. closed discrimination --------------------------------------
    closed_tip_avg_count: int = 3        # average N deepest pts for the closed tip
    # Open vs closed: the two blade tips are the contour points furthest into the
    # frame along the two diagonal directions (axis ± perp). If their separation
    # exceeds open_min_gap_px the tweezer is "open"; below it the blades have
    # converged ("closed"). open_two_blob_min_ratio decides whether a second
    # blob (separated blade) is merged into the point set before measuring.
    open_min_gap_px: float = 38.0
    open_two_blob_min_ratio: float = 0.20  # 2nd blob area >= ratio * 1st blob area
    # Split the forward tip region into the two blade branches, then detect one
    # representative tip point per branch. ``lead_slab_px`` controls how much of
    # the forward region participates in that branch split; ``tip_avg_depth_px``
    # is the averaging depth from each branch's leading edge used to stabilise
    # the final tip point under defocus.
    lead_slab_px: float = 140.0
    tip_avg_depth_px: float = 6.0
    # When the tweezer is open, fit one axis line to each blade branch using a
    # slightly deeper forward slice. Their intersection estimates where the tips
    # will meet once the blades close.
    line_fit_depth_px: float = 220.0
    open_fallback_forward_ratio: float = 0.18
    support_bin_width_px: float = 8.0
    support_front_exclude_px: float = 8.0
    open_forward_max_px: float = 8.0
    # The reported tip is the SEAM END: the centre of the mask's forward tip
    # region (the groove where the two blades converge), i.e. where the tip lands
    # once the blades close — not the protruding single blade tip. tip_slab_px is
    # how deep (along the inward axis) that forward region is averaged over.
    min_area_px_defocus: int = 1500      # relaxed area floor (faint defocused blade)
    tip_slab_px: float = 45.0
    # (legacy, unused now)
    open_notch_depth_px: float = 8.0
    tip_region_frac: float = 0.55

    # ---- open-tweezer → predicted CLOSED-tip point ---------------------------
    # When the tweezer is open we do NOT mark a point on the line between the two
    # tips; we mark where the tip will be AFTER the blades close. Calibrated from
    # the closed/open reference pair (same arm pose): the closed tip is at a
    # fraction ``open_pick_ratio`` along the upper→lower tip line, then pushed
    # ``open_forward_ratio * gap`` forward (perpendicular to the tip line, into
    # the frame). Both are scale-invariant, so they transfer between the 911px
    # reference and full-res frames. With gap→0 (closed) this collapses to the tip.
    open_pick_ratio: float = 0.436       # along upper→lower tip line (0=upper tip)
    open_forward_ratio: float = 0.124    # forward offset as a fraction of the tip gap
    open_ratio_from: str = "top"         # (legacy, unused)
    # Defocused open-blade fronts can smear backward/downward, especially on the
    # lower branch. Keep the front-end refinement anchored to the support-line
    # extrapolation so one fuzzy branch does not drag the pair geometry away.
    open_tip_max_backtrack_px: float = 18.0
    open_tip_max_lateral_drift_px: float = 18.0
    open_contour_tip_line_dist_weight: float = 0.5
    open_contour_tip_topk: int = 12
    open_contour_tip_front_band_px: float = 20.0
    open_contour_tip_darkness_weight: float = 1.2

    # ---- temporal stabilisation ----------------------------------------------
    ema_alpha: float = 0.5               # 1.0 = no smoothing, lower = smoother
    max_jump_px: float = 280.0           # ignore teleporting tips, hold previous
    miss_hold_frames: int = 3            # keep last tip for N misses before drop

    # ---- distance scaling -----------------------------------------------------
    mm_per_pixel: float = 0.009857       # telecentric lens scale (rig default)

    # ---- drawing --------------------------------------------------------------
    cross_color_bgr: Tuple[int, int, int] = (255, 0, 255)  # vivid magenta
    cross_size: int = 18                 # matches the foreign-object pick cross
    cross_thickness: int = 2

    # ---- debugging ------------------------------------------------------------
    debug_dump_dir: Optional[str] = None  # if set, dump mask/overlay PNGs

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TweezerConfig":
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {}
        for k, v in data.items():
            if k not in known:
                logger.warning("Unknown tweezer config key ignored: %s", k)
                continue
            if k == "cross_color_bgr" and isinstance(v, (list, tuple)):
                v = tuple(int(c) for c in v)
            kwargs[k] = v
        return cls(**kwargs)


@dataclass
class TweezerResult:
    """One frame's tweezer detection outcome (original-frame pixel coords)."""

    found: bool = False
    is_open: Optional[bool] = None       # True=open, False=closed, None=unknown
    tip_xy: Optional[Point] = None       # the reported pick/tip point
    tips: List[Point] = field(default_factory=list)  # 1 (closed) or 2 (open) blade tips
    entry_side: Optional[str] = None
    confidence: float = 0.0
    held: bool = False                   # True when reusing a previous detection

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "is_open": self.is_open,
            "state": (None if self.is_open is None else ("open" if self.is_open else "closed")),
            "tip_xy": list(self.tip_xy) if self.tip_xy else None,
            "tips": [list(t) for t in self.tips],
            "entry_side": self.entry_side,
            "confidence": round(self.confidence, 3),
            "held": self.held,
        }


_AXIS_INTO_FRAME = {
    # unit vector pointing from the entry edge *into* the image
    "left": (1.0, 0.0),
    "right": (-1.0, 0.0),
    "top": (0.0, 1.0),
    "bottom": (0.0, -1.0),
}


@dataclass
class _BranchSupportModel:
    upper_m: float
    upper_b: float
    lower_m: float
    lower_b: float
    eval_along: float
    support_upper_tip: Point
    support_lower_tip: Point


class TweezerDetector:
    """Stateful (for temporal smoothing) classical-CV tweezer tip detector."""

    def __init__(self, config: Optional[TweezerConfig] = None) -> None:
        self.config = config or TweezerConfig()
        self._prev_tip: Optional[Point] = None
        self._miss_streak: int = 0
        self._last_result: TweezerResult = TweezerResult(found=False)
        self._dbg_idx = 0

    # -- public API ------------------------------------------------------------

    def detect(self, image_bgr: np.ndarray) -> TweezerResult:
        """Locate the tweezer tip in one BGR frame."""
        cfg = self.config
        if image_bgr is None or image_bgr.size == 0:
            return self._on_miss()

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if cfg.blur_ksize and cfg.blur_ksize > 1:
            k = cfg.blur_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        mask = self._segment(gray)

        h, w = mask.shape[:2]

        # Entry side: fixed if configured, else inferred from the largest
        # border-touching blob. Candidates must touch THAT side — this is what
        # separates the tweezer (enters from one edge) from interior debris /
        # feathers that happen to be dark and sizeable.
        side = cfg.entry_side if cfg.entry_side in _AXIS_INTO_FRAME else None
        cands = self._select_candidates(mask, h, w, side)
        if not cands:
            self._maybe_dump(image_bgr, mask, None)
            return self._on_miss()
        if side is None:
            side = self._infer_entry_side(cands[0][0], h, w)
            cands = [c for c in cands if self._touches_side(c[0], h, w, side)] or cands
        axis = np.array(_AXIS_INTO_FRAME[side], dtype=float)

        result = self._classify(cands, mask.shape[:2], axis, side, gray)
        if result is None:
            self._maybe_dump(image_bgr, mask, None)
            return self._on_miss()

        # temporal smoothing + jump rejection
        tip = result.tip_xy
        assert tip is not None
        if self._prev_tip is not None:
            jump = math.dist(tip, self._prev_tip)
            if jump > cfg.max_jump_px and self._miss_streak == 0:
                # likely a spurious detection — keep the previous tip this frame
                logger.debug("tweezer tip jump %.0fpx > %.0f, holding previous", jump, cfg.max_jump_px)
                result.tip_xy = self._prev_tip
                result.held = True
            else:
                a = float(np.clip(cfg.ema_alpha, 0.05, 1.0))
                sx = a * tip[0] + (1 - a) * self._prev_tip[0]
                sy = a * tip[1] + (1 - a) * self._prev_tip[1]
                result.tip_xy = (sx, sy)

        self._prev_tip = result.tip_xy
        self._miss_streak = 0
        self._last_result = result
        self._maybe_dump(image_bgr, mask, result)
        return result

    def reset(self) -> None:
        """Forget temporal state (call when the scene changes wholesale)."""
        self._prev_tip = None
        self._miss_streak = 0
        self._last_result = TweezerResult(found=False)

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        result: TweezerResult,
        pick_points_xy: Optional[Sequence[Point]] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """Draw the tweezer cross (and tip↔pick distance) onto ``frame_bgr``.

        Returns the frame and the nearest-pick distance in **millimetres**
        (``None`` if no tip or no pick points). Draws in place and also returns
        the frame for chaining.
        """
        cfg = self.config
        if not result.found or result.tip_xy is None:
            return frame_bgr, None

        color = tuple(int(c) for c in cfg.cross_color_bgr)
        tip = (int(round(result.tip_xy[0])), int(round(result.tip_xy[1])))

        # Open state: show the two detected blade tips (small hollow dots) and a
        # faint line between them, so the predicted closed-tip cross is clearly
        # the convergence point — not one of the tips.
        if len(result.tips) == 2:
            t0 = tuple(int(round(v)) for v in result.tips[0])
            t1 = tuple(int(round(v)) for v in result.tips[1])
            cv2.line(frame_bgr, t0, t1, color, 1, cv2.LINE_AA)
            for t in (t0, t1):
                cv2.circle(frame_bgr, t, 6, color, 2, cv2.LINE_AA)

        # The tip marker — same MARKER_CROSS size/thickness as the seg pick cross.
        cv2.drawMarker(
            frame_bgr, tip, color,
            markerType=cv2.MARKER_CROSS,
            markerSize=cfg.cross_size,
            thickness=cfg.cross_thickness,
        )
        state_txt = "OPEN" if result.is_open else "CLOSED"
        self._label(frame_bgr, f"TWEEZER {state_txt}", (tip[0] + 12, tip[1] - 12), color)

        # Distance to nearest predicted pick point (real mm).
        dist_mm: Optional[float] = None
        if pick_points_xy:
            nearest = min(pick_points_xy, key=lambda p: math.dist(result.tip_xy, p))  # type: ignore[arg-type]
            d_px = math.dist(result.tip_xy, nearest)
            dist_mm = d_px * cfg.mm_per_pixel
            np_i = (int(round(nearest[0])), int(round(nearest[1])))
            cv2.line(frame_bgr, tip, np_i, color, 1, cv2.LINE_AA)
            mid = ((tip[0] + np_i[0]) // 2, (tip[1] + np_i[1]) // 2)
            self._label(frame_bgr, f"d={dist_mm:.2f}mm", (mid[0] + 8, mid[1]), color)

        return frame_bgr, dist_mm

    # -- internals -------------------------------------------------------------

    def _on_miss(self) -> TweezerResult:
        cfg = self.config
        self._miss_streak += 1
        if self._prev_tip is not None and self._miss_streak <= cfg.miss_hold_frames:
            held = TweezerResult(
                found=True,
                is_open=self._last_result.is_open,
                tip_xy=self._prev_tip,
                tips=self._last_result.tips,
                entry_side=self._last_result.entry_side,
                confidence=self._last_result.confidence * 0.5,
                held=True,
            )
            return held
        self._prev_tip = None
        self._last_result = TweezerResult(found=False)
        return self._last_result

    def _segment(self, gray: np.ndarray) -> np.ndarray:
        """Return a binary uint8 mask of the metal tweezer body."""
        cfg = self.config
        if cfg.seg_method == "dark_adaptive":
            bg = float(np.percentile(gray, cfg.bg_percentile))
            thr = max(1.0, bg - cfg.dark_margin)
            _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        elif cfg.seg_method == "dark":
            _, mask = cv2.threshold(gray, int(cfg.dark_threshold), 255, cv2.THRESH_BINARY_INV)
        elif cfg.seg_method == "bright":
            if cfg.use_otsu:
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, mask = cv2.threshold(gray, int(cfg.bright_threshold), 255, cv2.THRESH_BINARY)
        else:
            # edge-magnitude segmentation (default): strong gradients = metal.
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
            _, mask = cv2.threshold(mag, int(cfg.edge_threshold), 255, cv2.THRESH_BINARY)
            if cfg.edge_close_ksize and cfg.edge_close_ksize > 1:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.edge_close_ksize,) * 2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        if cfg.morph_open_ksize and cfg.morph_open_ksize > 1:
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_open_ksize,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
        if cfg.morph_close_ksize and cfg.morph_close_ksize > 1:
            kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close_ksize,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
        return mask

    def _select_candidates(self, mask: np.ndarray, h: int, w: int, side):
        """Return [(contour, area), ...] sorted by area desc, area-filtered and
        (if ``side`` is set) required to touch that entry border."""
        cfg = self.config
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_area = cfg.max_area_ratio * h * w
        area_floor = min(cfg.min_area_px, cfg.min_area_px_defocus)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < area_floor or area > max_area:
                continue
            if side is not None:
                if not self._touches_side(c, h, w, side):
                    continue
            elif cfg.require_border_touch and not self._touches_border(c, h, w):
                continue
            out.append((c, area))
        out.sort(key=lambda t: -t[1])
        return out

    def _touches_border(self, cnt: np.ndarray, h: int, w: int) -> bool:
        b = self.config.border_band_px
        pts = cnt.reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        return bool(
            np.any(x <= b) or np.any(x >= w - 1 - b)
            or np.any(y <= b) or np.any(y >= h - 1 - b)
        )

    def _touches_side(self, cnt: np.ndarray, h: int, w: int, side: str) -> bool:
        b = self.config.border_band_px
        pts = cnt.reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        if side == "left":
            return bool(np.any(x <= b))
        if side == "right":
            return bool(np.any(x >= w - 1 - b))
        if side == "top":
            return bool(np.any(y <= b))
        if side == "bottom":
            return bool(np.any(y >= h - 1 - b))
        return False

    def _lead_tip(self, cnt: np.ndarray, axis: np.ndarray):
        """(tip_xy, pts, proj) — tip = mean of the N points deepest into frame."""
        pts = cnt.reshape(-1, 2).astype(float)
        proj = pts @ axis
        n = max(1, int(self.config.closed_tip_avg_count))
        idx = np.argsort(proj)[::-1][:n]
        tip = pts[idx].mean(axis=0)
        return (float(tip[0]), float(tip[1])), pts, proj

    def _infer_entry_side(self, cnt: np.ndarray, h: int, w: int) -> str:
        """Pick the border edge holding the most contour mass."""
        b = max(self.config.border_band_px, 4)
        pts = cnt.reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        counts = {
            "left": int(np.count_nonzero(x <= b)),
            "right": int(np.count_nonzero(x >= w - 1 - b)),
            "top": int(np.count_nonzero(y <= b)),
            "bottom": int(np.count_nonzero(y >= h - 1 - b)),
        }
        side = max(counts, key=counts.get)
        if counts[side] == 0:
            # Not actually touching a border — fall back to "deepest from centroid".
            return "right"
        return side

    def _split_tip_branches(
        self,
        pts: np.ndarray,
        axis: np.ndarray,
        depth_px: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Split the forward tip region into upper/lower blade branches."""
        proj = pts @ axis
        lead = pts[proj >= float(proj.max()) - float(depth_px)]
        if len(lead) < 8:
            return None
        perp = np.array([-axis[1], axis[0]], dtype=float)
        lat = (lead @ perp).astype(np.float32).reshape(-1, 1)
        if float(lat.max() - lat.min()) < 2.0:
            return None
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.2,
        )
        try:
            _, labels, _ = cv2.kmeans(
                lat,
                2,
                None,
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS,
            )
        except cv2.error:
            return None
        groups = [lead[labels.ravel() == i] for i in range(2)]
        if min(len(g) for g in groups) < 3:
            return None
        groups.sort(key=lambda g: float(g[:, 1].mean()))
        return groups[0], groups[1]

    def _branch_tip(self, pts: np.ndarray, axis: np.ndarray) -> Optional[Point]:
        if len(pts) < 3:
            return None
        proj = pts @ axis
        mx = float(proj.max())
        depth = max(1.0, float(self.config.tip_avg_depth_px))
        sel = pts[proj >= mx - depth]
        if len(sel) == 0:
            sel = pts[np.argsort(proj)[::-1][:3]]
        tip = sel.mean(axis=0)
        return float(tip[0]), float(tip[1])

    def _filled_candidate_points(self, cands, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [cands[0][0]], -1, 255, -1)
        if len(cands) >= 2 and cands[1][1] >= self.config.open_two_blob_min_ratio * cands[0][1]:
            cv2.drawContours(mask, [cands[1][0]], -1, 255, -1)
        ys, xs = np.where(mask > 0)
        return np.column_stack([xs, ys]).astype(float)

    def _candidate_contour_points(self, cands) -> np.ndarray:
        contours = [cands[0][0]]
        if (
            len(cands) >= 2
            and cands[1][1] >= self.config.open_two_blob_min_ratio * cands[0][1]
        ):
            contours.append(cands[1][0])
        return np.concatenate(
            [cnt.reshape(-1, 2).astype(float) for cnt in contours],
            axis=0,
        )

    def _fit_support_model(self, pts: np.ndarray, axis: np.ndarray) -> Optional[_BranchSupportModel]:
        """Fit two branch centre lines in the blade support region."""
        if len(pts) < 40:
            return None
        cfg = self.config
        perp = np.array([-axis[1], axis[0]], dtype=float)
        along = pts @ axis
        lat = pts @ perp
        pmax = float(along.max())
        support_depth = max(float(cfg.line_fit_depth_px), 40.0)
        front_exclude = max(float(cfg.support_front_exclude_px), 2.0)
        bin_width = max(float(cfg.support_bin_width_px), 4.0)
        lower = pmax - support_depth
        upper = pmax - front_exclude
        if upper <= lower:
            return None

        bins: list[tuple[float, float, float]] = []
        start = lower
        while start < upper:
            sel = (along >= start) & (along < start + bin_width)
            chunk_lat = lat[sel]
            if len(chunk_lat) < 12:
                start += bin_width
                continue
            med = float(np.median(chunk_lat))
            upper_branch = chunk_lat[chunk_lat >= med]
            lower_branch = chunk_lat[chunk_lat < med]
            if len(upper_branch) < 4 or len(lower_branch) < 4:
                start += bin_width
                continue
            bins.append((
                start + 0.5 * bin_width,
                float(upper_branch.mean()),
                float(lower_branch.mean()),
            ))
            start += bin_width

        if len(bins) < 4:
            return None

        fit_x = np.array([[b[0], 1.0] for b in bins], dtype=float)
        upper_y = np.array([b[1] for b in bins], dtype=float)
        lower_y = np.array([b[2] for b in bins], dtype=float)
        m_up, b_up = np.linalg.lstsq(fit_x, upper_y, rcond=None)[0]
        m_lo, b_lo = np.linalg.lstsq(fit_x, lower_y, rcond=None)[0]

        eval_along = pmax - front_exclude
        up_lat = float(m_up * eval_along + b_up)
        lo_lat = float(m_lo * eval_along + b_lo)
        tip_a = axis * eval_along + perp * up_lat
        tip_b = axis * eval_along + perp * lo_lat
        pt_a = (float(tip_a[0]), float(tip_a[1]))
        pt_b = (float(tip_b[0]), float(tip_b[1]))
        ordered = self._order_tips(pt_a, pt_b, axis)
        return _BranchSupportModel(
            upper_m=float(m_up),
            upper_b=float(b_up),
            lower_m=float(m_lo),
            lower_b=float(b_lo),
            eval_along=float(eval_along),
            support_upper_tip=ordered[0],
            support_lower_tip=ordered[1],
        )

    def _support_tip_pair(self, pts: np.ndarray, axis: np.ndarray) -> Optional[Tuple[Point, Point]]:
        model = self._fit_support_model(pts, axis)
        if model is None:
            return None
        return model.support_upper_tip, model.support_lower_tip

    def _refine_open_tips(
        self,
        pts: np.ndarray,
        axis: np.ndarray,
        model: _BranchSupportModel,
    ) -> Optional[Tuple[Point, Point]]:
        """Use real front-end pixels to recover each branch tip."""
        cfg = self.config
        perp = np.array([-axis[1], axis[0]], dtype=float)
        along = pts @ axis
        lat = pts @ perp
        pmax = float(along.max())
        tip_region = pts[along >= pmax - max(float(cfg.lead_slab_px), 20.0)]
        if len(tip_region) < 10:
            return None

        tip_along = tip_region @ axis
        tip_lat = tip_region @ perp
        upper_pred = model.upper_m * tip_along + model.upper_b
        lower_pred = model.lower_m * tip_along + model.lower_b
        upper_dist = np.abs(tip_lat - upper_pred)
        lower_dist = np.abs(tip_lat - lower_pred)
        upper_pts = tip_region[upper_dist <= lower_dist]
        lower_pts = tip_region[lower_dist < upper_dist]
        if len(upper_pts) < 4 or len(lower_pts) < 4:
            return None

        tip_upper = self._branch_tip(upper_pts, axis)
        tip_lower = self._branch_tip(lower_pts, axis)
        if tip_upper is None or tip_lower is None:
            return None
        return self._order_tips(tip_upper, tip_lower, axis)

    def _stabilize_refined_tips(
        self,
        refined_pair: Tuple[Point, Point],
        model: Optional[_BranchSupportModel],
        axis: np.ndarray,
    ) -> Tuple[Point, Point]:
        """Clamp branch-tip refinement against support-line drift.

        In heavily defocused open frames, the front blur can make one branch's
        front slice retreat noticeably backward/downward relative to the support
        model fitted on the clearer blade body. We keep the refined point, but
        limit how far it may walk away from the support extrapolation.
        """
        if model is None:
            return refined_pair

        cfg = self.config
        perp = np.array([-axis[1], axis[0]], dtype=float)
        support_pair = (model.support_upper_tip, model.support_lower_tip)
        max_backtrack = max(0.0, float(cfg.open_tip_max_backtrack_px))
        max_lateral = max(0.0, float(cfg.open_tip_max_lateral_drift_px))
        stabilized: list[Point] = []

        for refined_tip, support_tip in zip(refined_pair, support_pair):
            refined = np.array(refined_tip, dtype=float)
            support = np.array(support_tip, dtype=float)
            delta = refined - support

            retreat = float(support @ axis) - float(refined @ axis)
            if retreat > max_backtrack > 0.0:
                alpha = max_backtrack / retreat
                refined = support + alpha * delta
                delta = refined - support

            lateral = abs(float(delta @ perp))
            if lateral > max_lateral > 0.0:
                alpha = max_lateral / lateral
                refined = support + alpha * delta

            stabilized.append((float(refined[0]), float(refined[1])))

        return self._order_tips(stabilized[0], stabilized[1], axis)

    def _predict_closed_tip_from_pair(
        self,
        tip_upper: Point,
        tip_lower: Point,
        axis: np.ndarray,
        *,
        is_open: bool,
    ) -> Point:
        """Predict the closed-tip point from the two branch tips."""
        midpoint = (
            (tip_upper[0] + tip_lower[0]) * 0.5,
            (tip_upper[1] + tip_lower[1]) * 0.5,
        )
        if not is_open:
            return midpoint

        cfg = self.config
        ratio = float(np.clip(cfg.open_pick_ratio, 0.0, 1.0))
        base = (
            tip_upper[0] + (tip_lower[0] - tip_upper[0]) * ratio,
            tip_upper[1] + (tip_lower[1] - tip_upper[1]) * ratio,
        )
        gap = math.dist(tip_upper, tip_lower)
        forward_px = min(
            float(cfg.open_forward_max_px),
            max(0.0, float(cfg.open_forward_ratio) * gap),
        )
        return (
            base[0] + forward_px * float(axis[0]),
            base[1] + forward_px * float(axis[1]),
        )

    def _should_fallback_open_lower_to_support(
        self,
        support_lower: Point,
        refined_lower: Optional[Point],
        contour_lower: Optional[Point],
    ) -> bool:
        """Detect the rare severe-defocus case where the lower open tip is unstable.

        Symptom pattern seen in the bad frame:
        - contour-based lower tip gets pulled *above* the support-line tip
        - fill-based refined lower tip gets pulled *below* the support-line tip
        - the two disagree by a large margin

        In that case the support-line extrapolation is the least-bad estimate and
        is far more stable across the rest of the image set.
        """
        if refined_lower is None or contour_lower is None:
            return False
        support = np.array(support_lower, dtype=float)
        refined = np.array(refined_lower, dtype=float)
        contour = np.array(contour_lower, dtype=float)
        if contour[1] >= support[1]:
            return False
        if refined[1] <= support[1]:
            return False
        return float(np.linalg.norm(refined - contour)) >= 30.0

    def _branch_contour_tip(
        self,
        contour_pts: np.ndarray,
        axis: np.ndarray,
        model: _BranchSupportModel,
        *,
        branch: str,
        gray: Optional[np.ndarray] = None,
    ) -> Optional[Point]:
        if len(contour_pts) < 8:
            return None
        cfg = self.config
        perp = np.array([-axis[1], axis[0]], dtype=float)
        along = contour_pts @ axis
        lat = contour_pts @ perp
        upper_pred = model.upper_m * along + model.upper_b
        lower_pred = model.lower_m * along + model.lower_b
        upper_dist = np.abs(lat - upper_pred)
        lower_dist = np.abs(lat - lower_pred)

        if branch == "upper":
            branch_pts = contour_pts[upper_dist <= lower_dist]
            branch_along = along[upper_dist <= lower_dist]
            branch_dist = upper_dist[upper_dist <= lower_dist]
        else:
            branch_pts = contour_pts[lower_dist < upper_dist]
            branch_along = along[lower_dist < upper_dist]
            branch_dist = lower_dist[lower_dist < upper_dist]

        if len(branch_pts) < 4:
            return None

        front_band = max(4.0, float(cfg.open_contour_tip_front_band_px))
        front_limit = float(branch_along.max()) - front_band
        front_sel = branch_along >= front_limit
        branch_pts = branch_pts[front_sel]
        branch_along = branch_along[front_sel]
        branch_dist = branch_dist[front_sel]
        if len(branch_pts) < 4:
            return None

        weight = max(0.0, float(cfg.open_contour_tip_line_dist_weight))
        score = branch_along - weight * branch_dist
        if branch == "lower" and gray is not None:
            xy = np.round(branch_pts).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, gray.shape[1] - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, gray.shape[0] - 1)
            intensity = gray[xy[:, 1], xy[:, 0]].astype(float)
            bg = float(np.percentile(gray, cfg.bg_percentile))
            darkness = np.maximum(0.0, bg - intensity)
            score = score + float(cfg.open_contour_tip_darkness_weight) * darkness
        topk = max(4, int(cfg.open_contour_tip_topk))
        idx = np.argsort(score)[-topk:]
        tip = branch_pts[idx].mean(axis=0)
        return float(tip[0]), float(tip[1])

    def _fit_branch_line(self, pts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if len(pts) < 6:
            return None
        try:
            vx, vy, x0, y0 = cv2.fitLine(
                pts.astype(np.float32),
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            ).reshape(-1)
        except cv2.error:
            return None
        direction = np.array([float(vx), float(vy)], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return None
        return np.array([float(x0), float(y0)], dtype=float), direction / norm

    def _line_intersection(
        self,
        line_a: Tuple[np.ndarray, np.ndarray],
        line_b: Tuple[np.ndarray, np.ndarray],
    ) -> Optional[Point]:
        pa, da = line_a
        pb, db = line_b
        denom = da[0] * db[1] - da[1] * db[0]
        if abs(float(denom)) < 1e-4:
            return None
        delta = pb - pa
        ta = (delta[0] * db[1] - delta[1] * db[0]) / denom
        p = pa + ta * da
        if not np.all(np.isfinite(p)):
            return None
        return float(p[0]), float(p[1])

    def _classify(
        self,
        cands,
        shape: tuple[int, int],
        axis: np.ndarray,
        side: str,
        gray: Optional[np.ndarray] = None,
    ) -> Optional[TweezerResult]:
        """Detect two blade tips and derive the reported target from them.

        Closed state:
            target = midpoint of the two detected tip points.

        Open state:
            fit one axis line to each blade branch in the forward region and use
            their intersection as the predicted tip position after the blades
            close. This follows the user's definition more closely than the old
            "forward seam slab" heuristic and remains usable under mild defocus.
        """
        cfg = self.config
        big_c, big_a = cands[0]
        P = self._filled_candidate_points(cands, shape)
        if len(P) < 20:
            return None

        support_model = self._fit_support_model(P, axis)
        tip_pair = (
            None if support_model is None
            else (support_model.support_upper_tip, support_model.support_lower_tip)
        )
        if tip_pair is None:
            # Fallback for severe defocus: use the older forward-region split.
            tip_groups = self._split_tip_branches(P, axis, cfg.lead_slab_px)
            if tip_groups is None:
                return None
            upper_pts, lower_pts = tip_groups
            tip_upper = self._branch_tip(upper_pts, axis)
            tip_lower = self._branch_tip(lower_pts, axis)
            if tip_upper is None or tip_lower is None:
                return None
            tip_pair = self._order_tips(tip_upper, tip_lower, axis)

        tip_upper, tip_lower = tip_pair
        support_gap = math.dist(tip_upper, tip_lower)
        if support_model is not None and support_gap >= max(cfg.open_min_gap_px * 0.5, 14.0):
            refined_pair = self._refine_open_tips(P, axis, support_model)
            if refined_pair is not None:
                tip_upper, tip_lower = self._stabilize_refined_tips(
                    refined_pair, support_model, axis
                )
            contour_lower = self._branch_contour_tip(
                self._candidate_contour_points(cands),
                axis,
                support_model,
                branch="lower",
                gray=gray,
            )
            if contour_lower is not None:
                if self._should_fallback_open_lower_to_support(
                    support_model.support_lower_tip,
                    None if refined_pair is None else refined_pair[1],
                    contour_lower,
                ):
                    tip_lower = support_model.support_lower_tip
                else:
                    # Open lower branch: the contour-based tip is intentionally
                    # allowed to drift further from the support model than the
                    # generic stabilizer permits. The support model is fitted on
                    # the fuzzy blade body and systematically pulls the lower tip
                    # back into the gap on normal open frames.
                    tip_lower = contour_lower

        gap = math.dist(tip_upper, tip_lower)
        is_open = gap >= cfg.open_min_gap_px

        target_xy = self._predict_closed_tip_from_pair(
            tip_upper, tip_lower, axis, is_open=is_open
        )

        conf = 0.8 if not is_open else float(np.clip(gap / (cfg.open_min_gap_px * 2.0), 0.45, 1.0))
        return TweezerResult(
            found=True,
            is_open=is_open,
            tip_xy=target_xy,
            tips=[tip_upper, tip_lower],
            entry_side=side, confidence=conf,
        )

    def _deepest_leading_notch(
        self, cnt: np.ndarray, pts: np.ndarray, proj: np.ndarray, lead_thresh: float
    ):
        cfg = self.config
        try:
            hull = cv2.convexHull(cnt, returnPoints=False)
        except cv2.error:
            return None
        if hull is None or len(hull) < 4:
            return None
        try:
            defects = cv2.convexityDefects(cnt, hull)
        except cv2.error:
            return None
        if defects is None:
            return None
        best = None
        best_depth = cfg.open_notch_depth_px
        for row in defects[:, 0]:
            s, e, f, d = int(row[0]), int(row[1]), int(row[2]), float(row[3]) / 256.0
            if d < best_depth:
                continue
            # The notch floor (far point) AND both shoulders must sit in the
            # leading region — this rejects the rear pivot gap of the tweezer.
            if proj[f] < lead_thresh:
                continue
            if proj[s] < lead_thresh or proj[e] < lead_thresh:
                continue
            best = (s, e, d)
            best_depth = d
        return best

    def _order_tips(self, a: Point, b: Point, axis: np.ndarray) -> Tuple[Point, Point]:
        """Return (t0, t1) so that ``open_pick_ratio`` is measured from t0."""
        mode = self.config.open_ratio_from
        if mode in ("leading", "trailing"):
            pa = a[0] * axis[0] + a[1] * axis[1]
            pb = b[0] * axis[0] + b[1] * axis[1]
            leading_first = pa >= pb
            if mode == "trailing":
                leading_first = not leading_first
            return (a, b) if leading_first else (b, a)
        # top/bottom by image row (y)
        top_first = a[1] <= b[1]
        if mode == "bottom":
            top_first = not top_first
        return (a, b) if top_first else (b, a)

    def _label(self, frame: np.ndarray, text: str, org: Tuple[int, int], color) -> None:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def _maybe_dump(self, image, mask, result: Optional[TweezerResult]) -> None:
        d = self.config.debug_dump_dir
        if not d:
            return
        try:
            out = Path(d)
            out.mkdir(parents=True, exist_ok=True)
            self._dbg_idx = (self._dbg_idx + 1) % 100000
            stem = f"tw_{self._dbg_idx:05d}"
            cv2.imwrite(str(out / f"{stem}_mask.png"), mask)
            if result is not None and result.found:
                vis = image.copy()
                self.draw_overlay(vis, result, None)
                cv2.imwrite(str(out / f"{stem}_overlay.png"), vis)
        except Exception as exc:  # noqa: BLE001 — debug only
            logger.debug("tweezer debug dump failed: %s", exc)
