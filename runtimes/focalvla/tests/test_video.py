"""Frame-exact seeking, bounded caches and dataset batching without PNG files."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import av
import numpy as np
from focalvla.data import CacheDataset
from focalvla.video import CAMERAS, VideoReader, image_requests


def make_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=20)
        stream.width = stream.height = 64
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf':'18','g':'12','bf':'2'}
        for i in range(60):
            pixels = np.full((64,64,3), (i*3)%256, np.uint8)
            pixels[8:24, (i%30):(i%30)+12, 0] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]


class VideoTest(unittest.TestCase):
    def test_seek_order_duplicates_and_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'{i}.mp4' for i in range(3)]
            reference = [make_video(p) for p in paths]
            reader = VideoReader(max_open=1, cache_bytes=64*64*3*2, max_forward_gap=8)
            for sequence in ([42,1,43,1,59], [0,8,9,3], [59,58,59]):
                for j,p in enumerate(paths):
                    images = reader.read_many([(str(p),i) for i in sequence])
                    for image,i in zip(images,sequence): np.testing.assert_array_equal(image,reference[j][i])
                    self.assertLessEqual(reader.cached_bytes,reader.cache_limit)
                    self.assertLessEqual(len(reader.decoders),1)
            before = reader.decoded_frames
            reader.read_many([(str(paths[-1]),59)]*4)
            self.assertEqual(before,reader.decoded_frames)
            with self.assertRaises(IndexError): reader.read_many([(str(paths[0]),60)])
            reader.close()
            self.assertEqual(reader.cached_bytes,0)

    def test_dataset_needs_no_image_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared=root/'prepared'; prepared.mkdir()
            row={'id':0,'episode_index':0,'dataset_name':'dataset','length':40,'tasks':['Open the drawer.']}
            references=[]
            for cam in CAMERAS:
                key='videos/observation.images.'+cam
                row.update({key+'/chunk_index':0,key+'/file_index':0,key+'/from_timestamp':.5})
                path=root/'dataset'/f'{key}/chunk-000/file-000.mp4'
                references.append(make_video(path))
            (prepared/'episodes.json').write_text(json.dumps([row]))
            np.save(prepared/'records.npy', np.array([[0,0,0],[0,39,1],[0,17,2]],np.int32))
            cache=prepared/'cache/000000';cache.mkdir(parents=True)
            (cache/'COMPLETE.json').write_text(json.dumps({'goals':[{'text':'Open drawer.'}]}))
            arrays={'frames':np.array([0,39,17]),'state':np.zeros((40,16),np.float32),
                    'action':np.arange(40*12,dtype=np.float32).reshape(40,12),
                    'geo':np.zeros((3,2048),np.float32),'motion':np.zeros((3,256),np.float32),
                    'geo_valid':np.ones(3,bool),'motion_valid':np.ones(3,bool),'goal_ids':np.zeros(3,np.int32)}
            for name,value in arrays.items():np.save(cache/(name+'.npy'),value)
            dataset=CacheDataset(prepared,root,video_cache_mb=.05)
            indices=[1,0,2,1]
            rows=dataset.__getitems__(indices)
            for (sample,target),i in zip(rows,indices):
                frame=int(arrays['frames'][i])
                for cam in range(3):np.testing.assert_array_equal(sample['image'][cam],references[cam][10+frame])
                expected=arrays['action'][np.minimum(frame+np.arange(50),39)]
                np.testing.assert_array_equal(sample['action'],expected)
                self.assertEqual(sample['action'].shape,(50,12))
            self.assertFalse((cache/'images.bin').exists())
            self.assertFalse((cache/'offsets.npy').exists())
            self.assertIsNone(dataset.__getstate__()['records'])
            self.assertIsNone(dataset.__getstate__()['_episodes'])
            self.assertLess(len(pickle.dumps(dataset)),10000)
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertIsInstance(restored.records, np.memmap)
            np.testing.assert_array_equal(restored[1][0]['image'][0],rows[0][0]['image'][0])
            restored.close()
            dataset.close()


if __name__=='__main__': unittest.main()
