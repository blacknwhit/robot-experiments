"""Mmap teacher targets plus on-demand video frames, prefetched by DataLoader."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .common import read_json
from .video import CAMERAS, VideoReader, image_requests


class CacheDataset:
    def __init__(self, root, dataset_root, max_open=8, max_open_videos=16,
                 video_cache_mb=64, max_forward_gap=32):
        self.root = Path(root)
        self.dataset_root = Path(dataset_root)
        self.records = np.load(self.root/'records.npy', mmap_mode='r')
        self._episodes = None
        self.max_open = max_open
        self.open = OrderedDict()
        self.video_options = (max_open_videos, int(video_cache_mb*1024**2), max_forward_gap)
        self.video = None

    @property
    def episodes(self):
        if self._episodes is not None:
            return self._episodes
        keys = ['episode_index','dataset_name','length','tasks']
        keys += ['videos/observation.images.'+cam+'/'+field for cam in CAMERAS
                 for field in ('chunk_index','file_index','from_timestamp')]
        # The full LeRobot metadata also contains statistics used only offline.
        # Do not replicate those dictionaries into every training worker.
        self._episodes = [{k:row[k] for k in keys} for row in read_json(self.root/'episodes.json')]
        return self._episodes

    def __len__(self):
        return len(self.records)

    def _episode(self, index):
        if index in self.open:
            self.open.move_to_end(index)
            return self.open[index]
        p = self.root/'cache'/f'{index:06d}'
        metadata = read_json(p/'COMPLETE.json')
        values = {k: np.load(p/(k+'.npy'), mmap_mode='r') for k in
                  ('frames','state','action','geo','motion','geo_valid','motion_valid','goal_ids')}
        values['goals'] = metadata['goals']
        self.open[index] = values
        if len(self.open) > self.max_open:
            self.open.popitem(last=False)
        return values

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        # PyTorch's batched fetch hook allows one seek/ordered decode per file,
        # without changing the training sampler or the returned sample order.
        if self.video is None:
            self.video = VideoReader(*self.video_options)
        requests = []
        for index in indices:
            episode, frame, _ = map(int, self.records[index])
            requests.extend(image_requests(self.dataset_root, self.episodes[episode], frame))
        images = self.video.read_many(requests)
        return [self._sample(index, images[i*3:(i+1)*3]) for i,index in enumerate(indices)]

    def _sample(self, index, images):
        episode, frame, local = map(int, self.records[index])
        values = self._episode(episode)
        assert values['frames'][local] == frame
        goal = values['goals'][int(values['goal_ids'][local])]
        sample = {'image': images, 'state': np.array(values['state'][frame]),
                  'action': np.array(values['action'][np.minimum(frame+np.arange(50),len(values['action'])-1)]),
                  'lang': self.episodes[episode]['tasks'][0], 'subtask': goal['text']}
        targets = {k: np.array(values[k][local]) for k in ('geo','motion','geo_valid','motion_valid')}
        return sample, targets

    def close(self):
        if self.video is not None:
            self.video.close()
            self.video = None
        self.open.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['open'] = OrderedDict()
        state['video'] = None
        state['records'] = None  # np.memmap's default pickle copies the entire array.
        state['_episodes'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.records = np.load(self.root/'records.npy', mmap_mode='r')


class PlanBatches:
    """DataLoader prefetch does not change the committed optimizer-step cursor."""
    def __init__(self, root, start, stop, rank, world, microbatch, accumulation):
        self.draws = np.load(Path(root)/'draws.npy', mmap_mode='r')
        self.start, self.stop, self.rank, self.world = start, stop, rank, world
        self.microbatch, self.accumulation = microbatch, accumulation
        assert self.draws.shape[1] == world * microbatch * accumulation

    def __iter__(self):
        for step in range(self.start, self.stop):
            for micro in range(self.accumulation):
                offset = (micro*self.world + self.rank)*self.microbatch
                yield self.draws[step, offset:offset+self.microbatch].tolist()

    def __len__(self):
        return (self.stop-self.start)*self.accumulation

