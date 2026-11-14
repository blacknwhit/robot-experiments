"""Bounded, mmap-backed training reader. No teachers, simulator, or video seeks."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from .common import read_json


class CacheDataset:
    def __init__(self, root, max_open=8):
        self.root = Path(root)
        self.records = np.load(self.root/'records.npy', mmap_mode='r')
        self.episodes = read_json(self.root/'episodes.json')
        self.max_open = max_open
        self.open = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _episode(self, index):
        if index in self.open:
            self.open.move_to_end(index)
            return self.open[index]
        p = self.root/'cache'/f'{index:06d}'
        metadata = read_json(p/'COMPLETE.json')
        values = {k: np.load(p/(k+'.npy'), mmap_mode='r') for k in
                  ('frames','state','action','geo','motion','geo_valid','motion_valid','goal_ids','offsets')}
        values['rgb'] = (p/'images.bin').open('rb')
        values['goals'] = metadata['goals']
        self.open[index] = values
        if len(self.open) > self.max_open:
            _, old = self.open.popitem(last=False)
            old['rgb'].close()
        return values

    def __getitem__(self, index):
        import cv2
        episode, frame, local = map(int, self.records[index])
        values = self._episode(episode)
        assert values['frames'][local] == frame
        images = []
        for camera in range(3):
            lo, hi = values['offsets'][local, camera:camera+2]
            values['rgb'].seek(int(lo))
            encoded = np.frombuffer(values['rgb'].read(int(hi-lo)), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f'Unreadable image: episode={episode}, frame={frame}, camera={camera}')
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        goal = values['goals'][int(values['goal_ids'][local])]
        sample = {'image': images, 'state': np.array(values['state'][frame]),
                  'action': np.array(values['action'][np.minimum(frame+np.arange(50),len(values['action'])-1)]),
                  'lang': self.episodes[episode]['tasks'][0], 'subtask': goal['text']}
        targets = {k: np.array(values[k][local]) for k in ('geo','motion','geo_valid','motion_valid')}
        return sample, targets


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

