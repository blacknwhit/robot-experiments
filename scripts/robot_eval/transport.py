"""Same msgpack wire codec; explicit, connection-local episode lifecycle."""
import asyncio
import importlib.util
from pathlib import Path
import traceback


def codec():
    path=Path(__file__).resolve().parents[2]/'runtimes/baseline/deployment/model_server/tools/msgpack_numpy.py'
    spec=importlib.util.spec_from_file_location('_robot_eval_msgpack',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


async def serve(manager, host, port, *, ready=None, stop=None):
    from websockets.asyncio.server import serve as websocket_serve
    from websockets.exceptions import ConnectionClosed
    packing=codec()
    slots=asyncio.Semaphore(manager.adapter.max_concurrency)
    async def handler(ws):
        sid=manager.connect()
        try:
            await ws.send(packing.packb(manager.metadata))
            async for data in ws:
                async with slots:
                    result=await asyncio.to_thread(manager.dispatch,sid,packing.unpackb(data))
                await ws.send(packing.packb({'status':'ok','data':result}))
        except ConnectionClosed:pass
        except Exception:
            error=traceback.format_exc();print(error,flush=True)
            try:await ws.send(error)
            except ConnectionClosed:pass
            await ws.close(code=1011,reason='Model adapter failure')
        finally:
            # dispatch and cleanup share the session lock, including if cancelled.
            await asyncio.to_thread(manager.disconnect,sid)
    async with websocket_serve(handler,host,port,compression=None,max_size=None,ping_interval=None):
        if ready is not None:ready.set()
        await (stop.wait() if stop is not None else asyncio.Future())
