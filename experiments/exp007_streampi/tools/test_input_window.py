"""Fast CPU regression for nearest timestamps, partial action horizons and training hot path."""
import ast
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from input_pipeline import Episode, Windows, _video_window


class SyntheticCapture:
    def __init__(self, *args):
        self.pos = -1
    def isOpened(self):return True
    def get(self, key):return {1:30.0, 2:64}[key]
    def set(self, key, index):
        self.pos = index - 1
        return True
    def grab(self):
        self.pos += 1
        return self.pos < 64
    def retrieve(self):
        return True, np.asarray([[[self.pos, 0, 255]]], dtype=np.uint8)
    def release(self):pass


def timestamp_test():
    fake = types.SimpleNamespace(VideoCapture=SyntheticCapture,CAP_PROP_FPS=1,CAP_PROP_FRAME_COUNT=2,
                                 CAP_PROP_POS_FRAMES=3, COLOR_BGR2RGB=4, cvtColor=lambda x, _:x[...,::-1])
    rng = np.random.default_rng(78)
    cases = [rng.uniform(-.01,2.2,size=3) for _ in range(160)]
    cases.extend(np.asarray([i/30,(i+.5)/30,(i+1)/30]) for i in range(63))
    cases += [np.asarray([0.,0.,0.]), np.asarray([1.2,1.2,1.2]), np.asarray([2.2,0.,-.1])]
    with patch.dict(sys.modules, cv2=fake):
        for times in cases:
            got = np.asarray(_video_window("memory", times))[:,0,0]
            expected = np.abs(np.arange(64)[:,None]/30.0-times).argmin(axis=0)
            np.testing.assert_array_equal(got[:,2],expected)
            assert np.all(got[:,0] == 255)
    return len(cases)


class Series:
    def __init__(self, values):self.values=values
    def to_numpy(self):return self.values
    def __eq__(self, other):return self.values == other


class MiniTable:
    def __init__(self, data):self.cols=data
    def __len__(self):return len(self.cols["episode_index"])
    def __getitem__(self,key):return Series(self.cols[key])
    def slice(self, l, n):return MiniTable({k:v[l:l+n] for k,v in self.cols.items()})
    def select(self, cols):return MiniTable({k:self.cols[k] for k in cols})
    def to_pandas(self):return self


def tail_test():
    length=61
    ar=np.arange(length*12,dtype=np.float32).reshape(length,12)
    st=np.arange(length*16,dtype=np.float32).reshape(length,16)
    table=MiniTable(dict(timestamp=np.arange(length,dtype=np.float32)/30,
                         episode_index=np.full(length,42), lang=np.zeros(length), state=st, action=ar))
    nested=types.SimpleNamespace(data=types.SimpleNamespace(table=table))
    reader=types.SimpleNamespace(_open_dataset=lambda:nested)
    w=Windows.__new__(Windows); w.cfg=dict(seed=42,history_frames=3,history_interval=5,action_horizon=50)
    w.prob=[1.]; w.tasks=[(0,"fixture",[1.],[Episode(42,0,length,(("cam0",0.),("cam1",0.),("cam2",0.)))])]
    w.bundles=[(reader,[("state",slice(0,16),True)],[("action",slice(0,6),True),("action",slice(6,12),False)],"lang",{0:"pick"})]
    images=types.SimpleNamespace(resize_with_pad=lambda im,h,w:im)
    with patch.dict(sys.modules, {"openpi_client.image_tools":images}), patch("input_pipeline._video_window",lambda path,ts:list(ts)):
        for anchor in range(length):
            s=w.sample(9,current=anchor)
            times=np.asarray([max(0,anchor-10),max(0,anchor-5),anchor])
            np.testing.assert_array_equal(s["times"],times)
            for u,t in zip(s["units"],times):
                np.testing.assert_array_equal(u["state"],st[t])
                assert u["lang"]=="pick"
                np.testing.assert_array_equal(u["image"],np.repeat(table.cols["timestamp"][t],3))
            acts=s["units"][-1]["action"]
            indices=np.arange(anchor,anchor+50)
            expected=ar[np.minimum(indices,length-1)].copy()
            expected[indices>=length,6:]=0.
            np.testing.assert_array_equal(acts,expected)
    return length


def check_entry():
    tree=ast.parse((EXP/"run.py").read_text(encoding="utf-8"))
    hot=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=="step")
    calls={ast.unparse(n.func) for n in ast.walk(hot) if isinstance(n,ast.Call)}
    assert not any(s in c for c in calls for s in ("fingerprint","sha256","tobytes","ThreadPool"))
    return sorted(c for c in calls if c in ("next","jax.device_put","jax.block_until_ready"))


if __name__=="__main__":
    print(dict(nearest_timestamp_cases=timestamp_test(),action_windows=tail_test(),training_calls=check_entry()))
