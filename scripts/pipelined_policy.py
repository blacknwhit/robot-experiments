"""Original batch1 math, overlapped CPU transforms, and one exclusive GPU owner."""
import asyncio
import concurrent.futures
import copy
import threading
import time

import numpy as np
import torch
from cuda_graph_policy import CudaGraphPolicy,copy_tree,signature


def stack_tree(items):
    first=items[0]
    if isinstance(first,dict):return {key:stack_tree([item[key] for item in items]) for key in first}
    return np.stack([np.asarray(item) for item in items],axis=0)


def map_tree(fn,value):
    if isinstance(value,dict):return {key:map_tree(fn,item) for key,item in value.items()}
    if isinstance(value,tuple):return tuple(map_tree(fn,item) for item in value)
    if isinstance(value,list):return [map_tree(fn,item) for item in value]
    return fn(value)


class PipelinedCudaGraphPolicy:
    def __init__(self,framework,*,verify_every_request=False,max_inflight=16):
        self.framework=framework;self.eager=framework.predict_action;self.graph=CudaGraphPolicy(framework)
        torch.cuda.synchronize(self.graph.device)
        self.verify_every_request=verify_every_request;self.requests=0;self.closed=False
        self.cpu=concurrent.futures.ThreadPoolExecutor(1,thread_name_prefix='pi05-prepare')
        self.gpu=concurrent.futures.ThreadPoolExecutor(1,thread_name_prefix='pi05-gpu')
        self.bound=threading.BoundedSemaphore(max_inflight);self.transform_lock=threading.Lock()

    def prepare(self,sample):
        with self.transform_lock:
            transformed=self.framework._input_transform(self.framework._sample_to_openpi(copy.deepcopy(sample),include_actions=False))
        return stack_tree([transformed])

    @torch.inference_mode()
    def execute(self,sample,batch):
        verify=self.verify_every_request or self.requests==0
        expected=None
        if verify:
            with self.transform_lock:expected=self.eager([sample])
        f=self.framework;g=self.graph
        transformed=map_tree(lambda value:torch.as_tensor(value).to(device=g.device),batch)
        if transformed.pop('actions',None) is not None:raise ValueError('Actions are not inference inputs')
        observation=f._model_module.Observation.from_dict(transformed)
        generator=torch.Generator(device=g.device).manual_seed(int(sample['__pi05_rng_seed']))
        noise=torch.randn((1,f.action_horizon,f._model_config.action_dim),generator=generator,device=g.device,dtype=torch.float32)
        if g.graph is None:g._capture(observation,noise)
        if signature(vars(observation))!=g.input_signature:raise ValueError('Pipeline input shape changed')
        copy_tree(vars(g.static_observation),vars(observation));g.static_noise.copy_(noise);g.graph.replay()
        output=f._output_transform({'state':transformed['state'][0].detach().cpu().numpy(),'actions':g.static_output[0].detach().cpu().numpy()})
        actions=np.asarray(output['actions'],dtype=np.float32)[None]
        if actions.shape!=(1,f.action_horizon,f.environment_action_dim) or not np.isfinite(actions).all():raise ValueError('Invalid pipeline actions')
        actual={'actions':actions}
        if verify and not np.array_equal(expected['actions'],actions):
            raise RuntimeError(f"Pipeline actions differ from eager: max_abs={np.max(np.abs(expected['actions']-actions))}")
        self.requests+=1
        return expected if self.verify_every_request else actual

    def predict_action(self,examples,**kwargs):
        if len(examples)!=1 or kwargs:raise ValueError('The pipeline requires native batch1 with unchanged inference options')
        with self.bound:
            if self.closed:raise RuntimeError('Pipeline closed')
            sample=examples[0]
            if '__pi05_rng_seed' not in sample:raise ValueError('Explicit per-request seed required')
            result=concurrent.futures.Future()
            prepared=self.cpu.submit(self.prepare,sample)
            def prepared_done(future):
                try:
                    work=self.gpu.submit(self.execute,sample,future.result())
                    def executed(f):
                        try:result.set_result(f.result())
                        except BaseException as error:result.set_exception(error)
                    work.add_done_callback(executed)
                except BaseException as error:result.set_exception(error)
            prepared.add_done_callback(prepared_done)
            return result.result()

    def close(self):
        self.closed=True;self.cpu.shutdown(wait=True);self.gpu.shutdown(wait=True)


async def bounded_async_handler(server,websocket):
    # The event loop admits at most 16 requests BEFORE dispatching executor work.
    from deployment.model_server.tools import msgpack_numpy
    import websockets
    import websockets.frames
    import traceback
    if not hasattr(server,'_pipeline_slots'):server._pipeline_slots=asyncio.Semaphore(16)
    packer=msgpack_numpy.Packer();await websocket.send(packer.pack(server._metadata))
    try:
        async for data in websocket:
            message=msgpack_numpy.unpackb(data);server._last_active=time.time()
            async with server._pipeline_slots:result=await asyncio.to_thread(server._route_message,message)
            await websocket.send(packer.pack(result))
    except websockets.ConnectionClosed:pass
    except Exception:
        await websocket.send(traceback.format_exc())
        await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,reason='Pipeline failure; see traceback')
        raise
