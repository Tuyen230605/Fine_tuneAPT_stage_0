import os
import sys
import torch
import logging
import importlib
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

class StderrSilencer:
    def __enter__(self):
        self.stderr_fileno = sys.stderr.fileno()
        self.saved_stderr = os.dup(self.stderr_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.stderr_fileno)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stderr, self.stderr_fileno)
        os.close(self.saved_stderr)
        self.devnull.close()


# Per-worker VideoDecoder LRU cache.
# Workers are persistent processes; caching decoders avoids re-opening the NFS
# file and re-parsing the MP4 container header on every __getitem__ call.
_DECODER_CACHE: "OrderedDict[str, object]" = OrderedDict()
_DECODER_CACHE_MAX = 32


def _get_decoder(video_path: str, device: str):
    """Return a cached VideoDecoder, or create and cache a new one."""
    from torchcodec.decoders import VideoDecoder
    key = f"{video_path}|{device}"
    if key in _DECODER_CACHE:
        _DECODER_CACHE.move_to_end(key)
        return _DECODER_CACHE[key]
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    _DECODER_CACHE[key] = decoder
    if len(_DECODER_CACHE) > _DECODER_CACHE_MAX:
        _DECODER_CACHE.popitem(last=False)  # evict LRU entry
    return decoder


def decode_video_frames_torchcodec(
    video_path: Union[Path, str],
    timestamps: List[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    if importlib.util.find_spec("torchcodec") is None:
        raise ImportError("torchcodec is required but not available.")

    video_path = str(video_path)
    with StderrSilencer():
        decoder = _get_decoder(video_path, device)
        metadata = decoder.metadata
        average_fps = metadata.average_fps
        frame_indices = [round(ts * average_fps) for ts in timestamps]
        frames_batch = decoder.get_frames_at(indices=frame_indices)
        frame_data = frames_batch.data.clone().cpu()
        pts_data = frames_batch.pts_seconds.clone().cpu()
        del frames_batch

    loaded_frames = [frame_data[i] for i in range(len(frame_data))]
    loaded_ts = pts_data.tolist()
    del frame_data, pts_data

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    closest_frames = closest_frames.type(torch.float32) / 255
    assert len(timestamps) == len(closest_frames)
    return closest_frames
