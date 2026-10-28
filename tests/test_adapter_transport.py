import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_model_adapters import manager,payload,TASK


@unittest.skipUnless(importlib.util.find_spec('websockets') and importlib.util.find_spec('msgpack'),'wire dependencies unavailable')
class WireTests(unittest.TestCase):
    def test_real_websocket_lifecycle_and_disconnect_cleanup(self):
        import socket
        from websockets.sync.client import connect
        from robot_eval.transport import serve,codec
        m=manager();ready=threading.Event();box={};errors=[]
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        async def run():
            box['loop']=asyncio.get_running_loop();box['stop']=asyncio.Event()
            await serve(m,'127.0.0.1',port,ready=ready,stop=box['stop'])
        def thread():
            try:asyncio.run(run())
            except BaseException as error:errors.append(error);ready.set()
        worker=threading.Thread(target=thread,daemon=True);worker.start()
        packing=codec()
        try:
            self.assertTrue(ready.wait(10));self.assertEqual(errors,[])
            with connect(f'ws://127.0.0.1:{port}',proxy=None) as client:
                self.assertEqual(packing.unpackb(client.recv()),m.metadata)
                def rpc(kind,data):
                    client.send(packing.packb({'type':kind,'payload':data}))
                    result=packing.unpackb(client.recv());self.assertEqual(result['status'],'ok')
                    return result['data']
                context={'run_id':'wire','task':TASK,'episode':2}
                rpc('adapter_begin',context)
                result=rpc('adapter_infer',payload(context,0))
                self.assertEqual(result['actions'].shape,(1,50,12))
                receipt=rpc('adapter_end',{'context':context,'requests':1,'completed':True})
                self.assertTrue(receipt['state_released'])
                rpc('adapter_begin',{**context,'episode':3})
        finally:
            if 'loop' in box:box['loop'].call_soon_threadsafe(box['stop'].set)
            worker.join(10);m.close()
        self.assertFalse(worker.is_alive());self.assertEqual(errors,[])
        self.assertEqual([e[2] for e in m.adapter.ended],['completed','connection_closed'])


if __name__=='__main__':unittest.main()
