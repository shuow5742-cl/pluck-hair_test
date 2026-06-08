"""Pick process for Track ID lifecycle and business state management.

This module implements the business domain logic for pickable targets.
It manages:
- Track ID assignment and lifecycle
- Phase state machine for pick workflow
- Pick counting and statistics
- Position-based confirmation for pick success
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stabilizer import StableTarget

logger = logging.getLogger(__name__)


class TargetState(StrEnum):
    """Target business state."""

    PENDING = "pending"  # Waiting to be picked
    PICKED = "picked"  # Successfully picked
    ABANDONED = "abandoned"  # Failed too many times, skipped


class Phase(StrEnum):
    """Business workflow phase."""

    INIT = "init"  # Initialization, Stabilizer warming up
    READY = "ready"  # Ready for pick request
    AWAITING_PICK = "awaiting_pick"  # Target dispatched, waiting for robot
    CONFIRMING = "confirming"  # PICK_DONE received, confirming disappearance
    DONE = "done"  # All targets processed


@dataclass(slots=True)
class TrackedTarget:
    """Business target with track ID and state."""

    track_id: int
    x: float  # Center x (pixels)
    y: float  # Center y (pixels)
    width: float  # Bbox width (pixels)
    height: float  # Bbox height (pixels)
    confidence: float
    object_type: str
    state: TargetState
    cluster_id: str | None = None
    world_x: float | None = None
    world_y: float | None = None
    u: float | None = None
    pick_attempts: int = 0


@dataclass(slots=True)
class TrackStats:
    """Statistics for current region (Task scope)."""

    region_picked: int  # Number picked in current region
    current_pending: int  # Number of pending targets


@dataclass(slots=True)
class PickResult:
    """Result of a pick attempt."""

    success: bool
    target_id: int
    message: str  # "picked" | "not_disappeared"


@dataclass(slots=True)
class PickProcessConfig:
    """Configuration for PickProcess."""

    init_stable_threshold: int = 10  # Fixed init window size (frames)
    confirm_window_frames: int = 10  # CONFIRMING phase window size (frames)
    match_distance_threshold: float = 30.0  # Max center distance for matching (pixels)
    match_size_ratio_threshold: float = 0.3  # Max size difference ratio (0.3 = 30%)
    max_pick_attempts: int = 3


class PickProcess:
    """Manages Track ID lifecycle and business state.

    Phase state machine:
        INIT -> READY -> AWAITING_PICK -> CONFIRMING -> READY/DONE

    Confirmation uses position matching instead of cluster_id tracking.
    """

    def __init__(self, config: PickProcessConfig | None = None) -> None:
        self.config = config or PickProcessConfig()
        self.phase = Phase.INIT

        # Target storage by track_id
        self._targets: dict[int, TrackedTarget] = {}

        # Track ID generator
        self._track_id_counter = itertools.count(1)

        # Init phase tracking (fixed window counter)
        self._init_frame_count = 0

        # Statistics
        self._region_picked = 0

        # CONFIRMING phase state
        self._dispatched_target_id: int | None = None
        self._machine_picked_id: int | None = None
        self._confirm_frames_remaining: int = 0
        self._last_pick_result: PickResult | None = None

    def update(self, stable_targets: list[StableTarget]) -> None:
        """Update with current stable targets from Stabilizer.

        Behavior depends on current phase:
        - INIT: Create targets from stable_targets, check for init completion
        - CONFIRMING: Check if picked target disappeared using position matching
        - Other phases: Update target positions
        """
        if self.phase == Phase.INIT:
            self._update_init_phase(stable_targets)
        elif self.phase == Phase.CONFIRMING:
            self._update_confirming_phase(stable_targets)
        else:
            self._update_target_positions(stable_targets)

    def get_next_target(self) -> TrackedTarget | None:
        """Get next target for picking.

        Transitions: READY -> AWAITING_PICK

        Returns:
            Target to pick, or None if no targets available.
        """
        if self.phase != Phase.READY:
            logger.warning("get_next_target: phase=%s (need READY)", self.phase.value)
            return None

        for target in self._targets.values():
            if target.state == TargetState.PENDING:
                target.pick_attempts += 1
                self._dispatched_target_id = target.track_id
                self.phase = Phase.AWAITING_PICK
                logger.info(
                    "get_next_target: returning track_id=%s attempt=%s, phase READY -> AWAITING_PICK",
                    target.track_id,
                    target.pick_attempts,
                )
                return target

        return None

    def on_pick_done(self, target_id: int) -> None:
        """Signal that robot has finished picking a specific target.

        Transitions: READY/AWAITING_PICK -> CONFIRMING
        """
        if self.phase not in (Phase.READY, Phase.AWAITING_PICK):
            logger.warning(
                "on_pick_done ignored: phase=%s (need READY/AWAITING_PICK), track_id=%s",
                self.phase.value,
                target_id,
            )
            return

        target = self._targets.get(target_id)
        if target is None:
            # Target already gone = already picked
            logger.info(
                "on_pick_done: track_id=%s already gone, counting as picked. total_picked=%s",
                target_id,
                self._region_picked + 1,
            )
            self._dispatched_target_id = None
            self._region_picked += 1
            self._last_pick_result = PickResult(
                success=True,
                target_id=target_id,
                message="picked",
            )
            self._check_done()
            return

        logger.info(
            "on_pick_done: track_id=%s, phase %s -> CONFIRMING", target_id, self.phase.value
        )
        self._machine_picked_id = target_id
        self._confirm_frames_remaining = self.config.confirm_window_frames
        self.phase = Phase.CONFIRMING

    def get_stats(self) -> TrackStats:
        """Get current statistics for Task scope."""
        pending_count = sum(1 for t in self._targets.values() if t.state == TargetState.PENDING)
        return TrackStats(
            region_picked=self._region_picked,
            current_pending=pending_count,
        )

    def get_all_targets(self) -> list[TrackedTarget]:
        """Get all tracked targets."""
        return list(self._targets.values())

    def get_last_pick_result(self) -> PickResult | None:
        """Get result of last pick attempt, then clear it."""
        result = self._last_pick_result
        self._last_pick_result = None
        return result

    def reset(self) -> None:
        """Reset all state for new work cycle."""
        self.phase = Phase.INIT
        self._targets.clear()
        self._track_id_counter = itertools.count(1)
        self._init_frame_count = 0
        self._region_picked = 0
        self._dispatched_target_id = None
        self._machine_picked_id = None
        self._confirm_frames_remaining = 0
        self._last_pick_result = None

    def get_dispatch_state(self, target: TrackedTarget) -> str:
        """Return JSON-RPC style dispatch state for a target."""
        if target.pick_attempts <= 1:
            return "new_target"
        if target.pick_attempts == 2:
            return "retry_1"
        return "retry_2"

    # ==================== Internal Methods ====================

    def _update_init_phase(self, stable_targets: list[StableTarget]) -> None:
        """Update during INIT phase: create targets on first frame, update positions after."""
        if self._init_frame_count == 0:
            # First frame: create all targets
            for stable in stable_targets:
                self._create_target(stable)
        else:
            # Subsequent frames: update positions
            self._update_target_positions(stable_targets)

        self._init_frame_count += 1
        if self._init_frame_count >= self.config.init_stable_threshold:
            self.phase = Phase.READY
            logger.info("INIT phase complete, phase -> READY, targets=%s", len(self._targets))

    def _update_confirming_phase(self, stable_targets: list[StableTarget]) -> None:
        """Check if picked target disappeared using position matching."""
        self._confirm_frames_remaining -= 1

        if self._machine_picked_id is None:
            self._finish_confirming()
            return

        target = self._targets.get(self._machine_picked_id)
        if target is None:
            # Target already removed = picked
            self._on_pick_confirmed()
            return

        # Position matching: find if target still exists
        match = self._find_matching_detection(target, stable_targets)

        if match is None:
            # Target not found = picked successfully
            target.state = TargetState.PICKED
            self._on_pick_confirmed()
        elif self._confirm_frames_remaining <= 0:
            # Window expired, target still there = pick failed
            self._on_pick_failed()
        else:
            # Update target position from match
            target.x = match.x
            target.y = match.y
            target.width = match.width
            target.height = match.height
            target.confidence = match.confidence
            target.cluster_id = match.cluster_id
            target.world_x = getattr(match, "world_x", None)
            target.world_y = getattr(match, "world_y", None)
            match_u = getattr(match, "u", None)
            if match_u is not None:
                target.u = match_u

    def _update_target_positions(self, stable_targets: list[StableTarget]) -> None:
        """Update target positions from stable_targets using position matching."""
        for target in self._targets.values():
            if target.state != TargetState.PENDING:
                continue
            match = self._find_matching_detection(target, stable_targets)
            if match is not None:
                target.x = match.x
                target.y = match.y
                target.width = match.width
                target.height = match.height
                target.confidence = match.confidence
                target.cluster_id = match.cluster_id
                target.world_x = getattr(match, "world_x", None)
                target.world_y = getattr(match, "world_y", None)
                match_u = getattr(match, "u", None)
                if match_u is not None:
                    target.u = match_u

    def _find_matching_detection(
        self,
        target: TrackedTarget,
        stable_targets: list[StableTarget],
    ) -> StableTarget | None:
        """Find detection matching target by position and size.

        Matching criteria:
        1. Euclidean distance < match_distance_threshold
        2. Size ratio within match_size_ratio_threshold

        Returns closest match or None if no match found.
        """
        best_match = None
        best_distance = float("inf")

        for det in stable_targets:
            # Distance check
            distance = math.hypot(det.x - target.x, det.y - target.y)
            if distance > self.config.match_distance_threshold:
                continue

            # Size similarity check
            w_ratio = abs(det.width - target.width) / max(target.width, 1.0)
            h_ratio = abs(det.height - target.height) / max(target.height, 1.0)
            if w_ratio > self.config.match_size_ratio_threshold:
                continue
            if h_ratio > self.config.match_size_ratio_threshold:
                continue

            if distance < best_distance:
                best_distance = distance
                best_match = det

        return best_match

    def _create_target(self, stable: StableTarget) -> TrackedTarget:
        """Create new tracked target from stable target."""
        track_id = next(self._track_id_counter)
        target = TrackedTarget(
            track_id=track_id,
            x=stable.x,
            y=stable.y,
            width=stable.width,
            height=stable.height,
            confidence=stable.confidence,
            object_type=stable.object_type,
            state=TargetState.PENDING,
            cluster_id=stable.cluster_id,
            world_x=getattr(stable, "world_x", None),
            world_y=getattr(stable, "world_y", None),
            u=getattr(stable, "u", None),
        )
        self._targets[track_id] = target
        return target

    def _on_pick_confirmed(self) -> None:
        """Handle successful pick confirmation."""
        self._dispatched_target_id = None
        self._region_picked += 1
        logger.info(
            "Pick confirmed: track_id=%s, total_picked=%s",
            self._machine_picked_id,
            self._region_picked,
        )
        self._last_pick_result = PickResult(
            success=True,
            target_id=self._machine_picked_id,
            message="picked",
        )
        self._machine_picked_id = None
        self._finish_confirming()

    def _on_pick_failed(self) -> None:
        """Handle failed pick (target not disappeared)."""
        target = (
            self._targets.get(self._machine_picked_id)
            if self._machine_picked_id is not None
            else None
        )
        message = "not_disappeared"
        if target is not None and target.pick_attempts >= self.config.max_pick_attempts:
            target.state = TargetState.ABANDONED
            message = "abandoned"
            logger.warning(
                "Target abandoned after %s attempts: track_id=%s",
                target.pick_attempts,
                target.track_id,
            )

        self._dispatched_target_id = None
        logger.warning(
            "Confirm window expired: track_id=%s not disappeared",
            self._machine_picked_id,
        )
        self._last_pick_result = PickResult(
            success=False,
            target_id=self._machine_picked_id,
            message=message,
        )
        self._machine_picked_id = None
        self._finish_confirming()

    def _finish_confirming(self) -> None:
        """Finish confirming phase and transition to next phase."""
        self._confirm_frames_remaining = 0
        self._check_done()

    def _check_done(self) -> None:
        """Check if all targets are done and transition phase."""
        has_pending = any(t.state == TargetState.PENDING for t in self._targets.values())
        if has_pending:
            self.phase = Phase.READY
        else:
            self.phase = Phase.DONE
            logger.info("All targets processed, phase -> DONE")
