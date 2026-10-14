"""Lazy OpenCV `VideoWriter` wrapper used to record episode rollouts."""

import os
import cv2
import numpy as np


class VideoWriter(object):
    """Frame resolution is inferred from the first `write()` call (lazy init).

    Output directory is created on demand. `finalize()` is idempotent.
    """

    def __init__(self, output_path: str, frame_rate: float):
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = None
        self.finalized = False
    
    def _lazy_init(self, bgr: np.ndarray):
        H, W = bgr.shape[:2]
        output_dir = os.path.dirname(self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.writer = cv2.VideoWriter(self.output_path, self.fourcc, 
                                      self.frame_rate, (W,  H))
    
    def write(self, bgr: np.ndarray):
        assert not self.finalized, "cannot call `write` after `finalize` is called"
        if self.writer is None:
            self._lazy_init(bgr)
        self.writer.write(bgr)
    
    def finalize(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.finalized = True

