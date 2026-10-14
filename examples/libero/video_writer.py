import os
import subprocess
import tempfile
import cv2
import numpy as np


class VideoWriter(object):
    def __init__(self, output_path: str, frame_rate: float):
        self.output_path = output_path
        self.frame_rate = frame_rate
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = None
        self.finalized = False
        # Write mp4v to a temp file, then re-encode to H.264 for VSCode compatibility
        self._tmp_path = output_path + ".tmp.mp4"

    def _lazy_init(self, bgr: np.ndarray):
        H, W = bgr.shape[:2]
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.writer = cv2.VideoWriter(self._tmp_path, self.fourcc,
                                      self.frame_rate, (W, H))

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
        if os.path.exists(self._tmp_path):
            subprocess.run(
                ["ffmpeg", "-y", "-i", self._tmp_path,
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "18", self.output_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True,
            )
            os.remove(self._tmp_path)

