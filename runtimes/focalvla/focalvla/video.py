"""On-demand CFR video decoding with bounded per-worker memory; no disk cache."""
from collections import OrderedDict, defaultdict
from pathlib import Path
import math

CAMERAS = ('robot0_agentview_left', 'robot0_eye_in_hand', 'robot0_agentview_right')


def image_requests(dataset_root, row, frame):
    if not 0 <= frame < row['length']:
        raise IndexError((row['episode_index'], frame, row['length']))
    result = []
    for camera in CAMERAS:
        key = 'videos/observation.images.' + camera
        path = Path(dataset_root)/row['dataset_name']/f"{key}/chunk-{row[key+'/chunk_index']:03d}/file-{row[key+'/file_index']:03d}.mp4"
        # LeRobot offsets are relative to the start of the shared video file.
        start = float(row[key+'/from_timestamp'])*20
        if not math.isfinite(start) or abs(start-round(start)) > .01:
            raise ValueError(f'Unaligned 20 Hz episode timestamp: {path}: {start}')
        result.append((str(path), int(round(start))+int(frame)))
    return result


class _Decoder:
    def __init__(self, path, max_forward_gap):
        import av
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        # Many DataLoader processes already provide parallelism. Avoid each
        # decoder spawning a machine-sized FFmpeg thread pool.
        self.stream.thread_count = 1
        if self.stream.average_rate is None or abs(float(self.stream.average_rate)-20) > .01:
            self.container.close()
            raise ValueError(f'Expected constant 20 Hz RGB video: {path}')
        self.origin = self.stream.start_time or 0
        self.tick_per_frame = 1/(20*self.stream.time_base)
        self.max_forward_gap = max_forward_gap
        self.iterator = None
        self.last = None
        self.path = path

    def read(self, frame_index):
        sequential = (self.iterator is not None and self.last is not None
                      and 0 < frame_index-self.last <= self.max_forward_gap)
        if not sequential:
            target = self.origin + int(round(frame_index*self.tick_per_frame))
            self.container.seek(target, stream=self.stream, backward=True, any_frame=False)
            self.iterator = iter(self.container.decode(self.stream))
        for frame in self.iterator:
            if frame.pts is None:
                raise ValueError(f'Missing presentation timestamp in {self.path}')
            index = int(round((frame.pts-self.origin)/self.tick_per_frame))
            self.last = index
            if index < frame_index:
                continue  # Decode references but do not convert unused frames to RGB.
            if index != frame_index:
                raise ValueError(f'Video frame missing: {self.path}: requested {frame_index}, decoded {index}')
            return frame.to_ndarray(format='rgb24')
        raise IndexError(f'Video ended before frame {frame_index}: {self.path}')

    def close(self):
        self.iterator = None
        self.container.close()


class VideoReader:
    """Group each batch by file/frame, then return images in original sample order.

    Only requested RGB frames are cached. Codec handles and decoded RGB have
    separate hard limits. Instances are created lazily inside DataLoader workers.
    """
    def __init__(self, max_open=16, cache_bytes=64*1024**2, max_forward_gap=32):
        if max_open < 1 or cache_bytes < 0 or max_forward_gap < 0:
            raise ValueError('Invalid video-reader resource limits')
        self.max_open = max_open
        self.cache_limit = cache_bytes
        self.max_forward_gap = max_forward_gap
        self.decoders = OrderedDict()
        self.frames = OrderedDict()
        self.cached_bytes = 0
        self.decoded_frames = 0
        self.cache_hits = 0

    def _decoder(self, path):
        if path in self.decoders:
            self.decoders.move_to_end(path)
            return self.decoders[path]
        if len(self.decoders) >= self.max_open:
            _, decoder = self.decoders.popitem(last=False)
            decoder.close()
        decoder = _Decoder(path, self.max_forward_gap)
        self.decoders[path] = decoder
        return decoder

    def _remember(self, key, image):
        if image.nbytes > self.cache_limit:
            return
        while self.frames and self.cached_bytes+image.nbytes > self.cache_limit:
            _, old = self.frames.popitem(last=False)
            self.cached_bytes -= old.nbytes
        self.frames[key] = image
        self.cached_bytes += image.nbytes

    def read_many(self, requests):
        result = {}
        missing = defaultdict(set)
        for path, index in requests:
            key = (str(path), int(index))
            if key in result:
                continue
            if key in self.frames:
                result[key] = self.frames[key]
                self.frames.move_to_end(key)
                self.cache_hits += 1
            else:
                missing[key[0]].add(key[1])
        for path, indices in missing.items():
            decoder = self._decoder(path)
            for index in sorted(indices):
                key = (path, index)
                image = decoder.read(index)
                self.decoded_frames += 1
                result[key] = image
                self._remember(key, image)
        return [result[str(path), int(index)] for path, index in requests]

    def close(self):
        for decoder in self.decoders.values():
            decoder.close()
        self.decoders.clear()
        self.frames.clear()
        self.cached_bytes = 0

