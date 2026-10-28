"""Use the existing benchmark observation/action conversion with optional RPC lifecycle."""
import threading
from . import API_VERSION


class EpisodeClient:
    def __init__(self, client, task, run_id):
        self.client,self.task,self.run_id=client,task,run_id
        self.context=None;self.requests=0;self.session_id=None
        self.lock=threading.RLock()

    def get_server_metadata(self):return self.client.get_server_metadata()

    def _rpc(self, kind, payload):
        response=self.client.predict_action({'type':kind,'payload':payload})
        if response.get('status')!='ok':raise RuntimeError('Adapter RPC failed: '+str(response))
        return response['data']

    def begin(self, episode):
        with self.lock:
            if self.context is not None:raise ValueError('Previous model episode still active')
            context={'run_id':self.run_id,'task':self.task,'episode':episode}
            receipt=self._rpc('adapter_begin',context)
            if receipt.get('api_version')!=API_VERSION or receipt.get('context')!=context or receipt.get('begun') is not True:
                raise ValueError('Invalid model begin receipt')
            self.context=context;self.requests=0;self.session_id=receipt['session_id']

    def predict_action(self, payload):
        with self.lock:
            if self.context is None:raise ValueError('No active model episode')
            result=self._rpc('adapter_infer',{**payload,'context':{**self.context,'replan':self.requests}})
            if result.get('adapter_request') != {'session_id':self.session_id,'replan':self.requests,'accepted':True}:
                raise ValueError('Foreign model request receipt')
            self.requests+=1
            return {'status':'ok','data':result}

    def end(self, completed):
        with self.lock:
            receipt=self._rpc('adapter_end',{'context':self.context,'requests':self.requests,'completed':completed})
            if (receipt.get('context')!=self.context or receipt.get('session_id')!=self.session_id
                    or receipt.get('requests')!=self.requests or receipt.get('state_released') is not True):
                raise ValueError('Invalid model end receipt')
            self.context=None
            return receipt

    def close(self):
        with self.lock:
            try:
                if self.context is not None:
                    self._rpc('adapter_abort',{'context':self.context})
                    self.context=None
            finally:self.client.close()


def create_policy(*, host, port, task_name, run_id):
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from examples.pi05_robocasa365.eval_files.model2pi05_robocasa365_interface import PolicyWrapper
    raw=WebsocketClientPolicy(host,port)
    try:
        metadata=raw.get_server_metadata()
        version=metadata.get('model_adapter_api_version')
        if version is not None and version!=API_VERSION:raise ValueError('Unsupported model adapter API')
        client=EpisodeClient(raw,task_name,run_id) if version is not None else raw
        wrapper=PolicyWrapper(host=host,port=port,task_name=task_name,base_seed=7,
                              strict_determinism=True,trace_determinism=False,client=client)
        if version is not None:
            wrapper.begin_model_episode=client.begin
            wrapper.end_model_episode=client.end
        return wrapper
    except BaseException:
        raw.close();raise
