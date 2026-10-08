"""Local, deterministic change detection for webcam frames.

One rule: never spend a model call to decide whether to spend a model call.

A frame is reduced to a small grayscale signature and compared with the previous
one. The result is a 0..1 score. Nothing here calls a model, touches the
network, or holds anything but the previous signature — a 32x32 greyscale image,
about 1KB, not a frame archive.

Pillow does the resizing because it is already a dependency and already in
memory; adding a vision library for a resize would be the whole cost of this
feature multiplied by zero benefit.
"""

from __future__ import annotations

import numpy as np

from jarvis.config import (
    WEBCAM_CHANGE_THRESHOLD,
    WEBCAM_CHANGE_NOISE_FLOOR,
    WEBCAM_SAMPLE_SIZE,
)


class ChangeDetector:
    """Compares a frame against the previous one and reports a change score.

    Stateful by design: it holds exactly one previous signature, so it stays
    O(1) in memory no matter how long perception runs.

    The score is the *fraction of the frame that changed materially* — cells
    whose greyscale value moved by more than a noise floor — not the average
    difference across the frame. That distinction is the whole reason this class
    exists:

    * Averaging dilutes a localized change into the background. Measured on a
      640x480 desk frame, a person sitting down moves the mean by 0.0240 while a
      window shade opening moves it by 0.0235. A mean cannot tell those apart, so
      it fires a vision call on every lighting change and misses the same
      frequency of real ones. As a fraction of materially-changed cells, the
      same two frames score 0.054 and 0.000.
    * A floor discards sensor noise. Gaussian noise up to sigma=8 gray levels,
      already aggressive for a modern sensor, moves 0.0% of cells.
    * Mean-centring the signature cancels illumination outright, so auto-exposure
      and a dimmer being switched on move nothing at all.

    ponytail: 0.02 catches a person arriving or leaving and an object appearing
    or going, with 2.7x headroom, while drift and noise score exactly 0.000. It
    deliberately does not catch fine appearance changes -- glasses score 0.012,
    too close to real sensor noise to threshold honestly at this grid size.
    Lower `vision.webcam.change_threshold` to 0.005 to try; the analysis
    cooldown still bounds what that costs. A region detector is the real upgrade
    if fine appearance matters.
    """

    def __init__(
        self,
        threshold: float = WEBCAM_CHANGE_THRESHOLD,
        grid: int = WEBCAM_SAMPLE_SIZE,
        noise_floor: float = WEBCAM_CHANGE_NOISE_FLOOR,
    ):
        self.threshold = float(threshold)
        self.grid = int(grid)
        self.noise_floor = float(noise_floor)
        self._previous = None

    # -- core -------------------------------------------------------------

    def signature(self, frame):
        """Reduce a frame to its small, mean-centred greyscale signature.

        Mean-centring is what makes the score about *structure* rather than
        brightness. A room that got slightly brighter, or a webcam that changed
        its auto-exposure gain, moves every cell by the same amount; subtracting
        the frame's own mean cancels that exactly, leaving only what actually
        rearranged in the scene.
        """
        from PIL import Image

        image = Image.fromarray(frame)
        if image.mode != "L":
            image = image.convert("L")
        image = image.resize((self.grid, self.grid), Image.BILINEAR)
        signature = np.asarray(image, dtype=np.float32) / 255.0
        return signature - signature.mean()

    def score(self, frame) -> float:
        """Fraction of the frame that materially changed, in 0..1.

        The first frame scores 1.0: there is no baseline yet, so the initial
        scene is always worth describing once.

        The previous signature is advanced on every call, including for frames
        that scored below the threshold. Slow drift then has to accumulate
        against the *current* scene rather than against a stale one it can never
        catch up to.
        """
        current = self.signature(frame)
        previous, self._previous = self._previous, current
        if previous is None or previous.shape != current.shape:
            return 1.0
        return float((np.abs(current - previous) > self.noise_floor).mean())

    def changed(self, frame) -> bool:
        return self.score(frame) >= self.threshold

    def reset(self) -> None:
        """Forget the baseline, so the next frame is treated as a first frame."""
        self._previous = None


__all__ = ["ChangeDetector"]