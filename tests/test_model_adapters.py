import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robot_eval.api import ModelAdapter,ManagedPolicy,request_seed
from robot_eval.client import EpisodeClient
from robot_eval.spec import load_spec,FORMAT

TASK='robocasa/CloseBlenderLid'
PROOF={'strict':True,'all_keys_present':True,'all_loaded_values_exact_at_model_dtype':True,'checkpoint_keys_dropped':0}


class Toy(ModelAdapter):
    state_mode='per_episode';rng_mode='explicit_per_request'
    load_proof=PROOF
    def __init__(self):self.ended=[];self.seen=[]
    def begin_episode(self,context):return {'count':0}
    def predict(self,request,state):
        state['count']+=1;self.seen.append((request.episode.episode,request.replan,request.seed))
        return np.full((50,12),state['count']+request.episode.episode,dtype=np.float32)
    def end_episode(self,context,state,reason):self.ended.append((context,state,reason))


def manager(adapter=None):
    adapter=adapter or Toy()
    ident={key:getattr(adapter,value) for key,value in {'inference_backend':'backend','inference_backend_version':'backend_version',
            'state_mode':'state_mode','rng_mode':'rng_mode','max_concurrency':'max_concurrency'}.items()}
    return ManagedPolicy(adapter,ident,{'CloseBlenderLid':900},seed_scope=lambda seed:nullcontext())


def begin(m,sid,episode=0):
    context={'run_id':'attempt','task':TASK,'episode':episode}
    m.dispatch(sid,{'type':'adapter_begin','payload':context})
    return context


def payload(context,replan):
    return {'context':{**context,'replan':replan},'examples':[{'image':[np.ones((256,256,3),np.uint8)]*3,
            'state':np.zeros(16,np.float32),'lang':'close the lid','robot_tag':'robot',
            '__pi05_rng_seed':request_seed(TASK,context['episode'],replan)}]}


class AdapterTests(unittest.TestCase):
    def test_interleaved_episode_state_and_metadata_are_isolated(self):
        m=manager();a=m.connect();b=m.connect();meta=copy.deepcopy(m.metadata)
        ca=begin(m,a);cb=begin(m,b,1)
        for sid,c,step,expected in [(a,ca,0,1),(b,cb,0,2),(a,ca,1,2),(b,cb,1,3)]:
            out=m.dispatch(sid,{'type':'adapter_infer','payload':payload(c,step)})
            np.testing.assert_array_equal(out['actions'],np.full((1,50,12),expected,np.float32))
        proof=m.dispatch(a,{'type':'adapter_end','payload':{'context':ca,'requests':2,'completed':True}})
        self.assertEqual(proof['requests'],2);self.assertTrue(proof['state_released'])
        self.assertEqual(m.metadata,meta);m.disconnect(b);m.close()
        self.assertEqual(len(m.adapter.ended),2)

    def test_duplicate_context_and_shared_state_rejected(self):
        m=manager();a=m.connect();b=m.connect();begin(m,a)
        with self.assertRaisesRegex(ValueError,'Duplicate'):begin(m,b)
        m.close()
        class Shared(Toy):
            state={}
            def begin_episode(self,context):return self.state
        m=manager(Shared());a=m.connect();b=m.connect();begin(m,a)
        with self.assertRaisesRegex(ValueError,'reused'):begin(m,b,1)
        self.assertEqual(m.adapter.ended,[]);m.close()

    def test_out_of_order_and_wrong_seed_fail_without_advancing(self):
        for change in ('replan','seed'):
            m=manager();sid=m.connect();c=begin(m,sid);p=payload(c,0)
            if change=='replan':p['context']['replan']=1
            else:p['examples'][0]['__pi05_rng_seed']+=1
            with self.assertRaises(ValueError):m.dispatch(sid,{'type':'adapter_infer','payload':p})
            with self.assertRaises(ValueError):m.dispatch(sid,{'type':'adapter_infer','payload':payload(c,0)})
            self.assertEqual(m.adapter.seen,[]);m.close()

    def test_invalid_actions_never_silently_truncated_or_cast(self):
        for value in (np.zeros((50,13),np.float32),np.zeros((50,12),np.float64),np.full((50,12),np.nan,np.float32)):
            toy=Toy();toy.predict=lambda request,state:value
            m=manager(toy);sid=m.connect();c=begin(m,sid)
            with self.assertRaisesRegex(ValueError,'finite float32'):m.dispatch(sid,{'type':'adapter_infer','payload':payload(c,0)})
            m.close()

    def test_outputs_own_model_buffer(self):
        toy=Toy();buffer=np.zeros((50,12),np.float32);toy.predict=lambda r,s:buffer
        m=manager(toy);sid=m.connect();c=begin(m,sid)
        out=m.dispatch(sid,{'type':'adapter_infer','payload':payload(c,0)})
        buffer[:]=9;self.assertEqual(float(out['actions'].sum()),0);m.close()

    def test_concurrency_limit_enforced_by_manager(self):
        class Concurrent(Toy):
            max_concurrency=2
            def __init__(self):super().__init__();self.active=0;self.maximum=0;self.mutex=threading.Lock()
            def predict(self,r,s):
                with self.mutex:self.active+=1;self.maximum=max(self.active,self.maximum)
                time.sleep(.03)
                with self.mutex:self.active-=1
                return super().predict(r,s)
        m=manager(Concurrent());work=[]
        for i in range(8):
            sid=m.connect();work.append((sid,begin(m,sid,i)))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda pair:m.dispatch(pair[0],{'type':'adapter_infer','payload':payload(pair[1],0)}),work))
        self.assertEqual(m.adapter.maximum,2);m.close()

    def test_cleanup_failure_quarantines_service_and_retains_state(self):
        toy=Toy()
        def fail(*args):raise RuntimeError('cleanup failed')
        toy.end_episode=fail;m=manager(toy);sid=m.connect();begin(m,sid)
        with self.assertRaises(RuntimeError):m.disconnect(sid)
        self.assertIsNotNone(m.sessions[sid].state)
        with self.assertRaises(ValueError):m.connect()
        toy.end_episode=lambda *args:None;m.close()

    def test_closed_service_rejects_new_connections_and_stale_dispatch(self):
        m=manager();sid=m.connect();m.disconnect(sid)
        with self.assertRaises(ValueError):begin(m,sid)
        m.close()
        with self.assertRaises(ValueError):m.connect()

    def test_client_lifecycle_uses_own_request_count_not_wrapper_counter(self):
        m=manager();sid=m.connect()
        class Raw:
            def predict_action(self,msg):return {'status':'ok','data':m.dispatch(sid,msg)}
            def get_server_metadata(self):return m.metadata
            def close(self):m.disconnect(sid)
        client=EpisodeClient(Raw(),TASK,'attempt');client.begin(3)
        p=payload(client.context,0);p.pop('context')
        out=client.predict_action(p);self.assertEqual(out['data']['actions'].shape,(1,50,12))
        receipt=client.end(True);self.assertEqual(receipt['context']['episode'],3)
        client.begin(4);client.close();self.assertFalse(m.active_states);m.close()


class SpecTests(unittest.TestCase):
    def test_source_asset_config_and_new_file_change_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'model';source.mkdir()
            for name in ('adapter.py','model.py'):(source/name).write_text('x=1\n')
            for name in ('config.json','weights.bin','norm.json'):(root/name).write_text('{}')
            data={'format':FORMAT,'model_id':'brand-new-architecture','factory':'adapter:create','config':'config.json',
                  'checkpoint':'weights.bin','python_paths':['model'],'source_roots':[{'path':'model'}],
                  'assets':{'normalization':'norm.json'}}
            path=root/'spec.json';path.write_text(json.dumps(data));spec=load_spec(path)
            first=spec.details({'python':'fixture'})['identity']
            (source/'new_module.py').write_text('x=2')
            second=spec.details({'python':'fixture'})['identity']
            self.assertNotEqual(first['model_sources_sha256'],second['model_sources_sha256'])
            (root/'norm.json').write_text('{"new":1}')
            third=spec.details({'python':'fixture'})['identity']
            self.assertNotEqual(second['model_assets_sha256'],third['model_assets_sha256'])
            self.assertNotEqual(second['model_spec_fingerprint'],third['model_spec_fingerprint'])
            self.assertEqual(spec.policy_environment(gpu='3')['CUDA_VISIBLE_DEVICES'],'3')
            self.assertEqual(spec.policy_environment(gpu='3')['JAX_PLATFORMS'],'cpu')
            self.assertEqual(spec.policy_environment(gpu='3')['NO_ALBUMENTATIONS_UPDATE'],'1')
            data['environment']={'CUDA_VISIBLE_DEVICES':'0'};path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_spec(path)


if __name__=='__main__':unittest.main()
