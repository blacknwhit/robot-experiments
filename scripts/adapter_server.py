"""Model-spec service. Does not import any benchmark model before its adapter."""
import argparse
import asyncio
import importlib
import inspect
import json
from pathlib import Path
import sys
from robot_eval.spec import load_spec,environment_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-spec',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    a=p.parse_args()
    expected=json.loads(a.manifest.read_text())
    spec=load_spec(a.model_spec,config=expected['config'],checkpoint=expected['checkpoint'])
    details=spec.details(environment_snapshot())
    if details != expected:raise ValueError('Policy model files/environment changed since orchestration')
    from robot_eval.api import LoadContext,ManagedPolicy
    from robot_eval.runtime import configure,metadata,seeded
    from robot_eval.transport import serve
    configure()
    paths=frozenset(spec.source_files().values())
    if spec.factory == 'pi05':
        from robot_eval.pi05 import create
    else:
        name,attribute=spec.factory.split(':')
        module=importlib.import_module(name)
        create=getattr(module,attribute)
        if Path(inspect.getfile(create)).resolve() not in paths:
            raise ValueError('Adapter factory not in declared source files')
    context=LoadContext(spec.config,spec.checkpoint,'cuda',spec.data['options'],spec.data['backend'],
                        spec.data['backend_version'],paths,spec.assets)
    adapter=create(context)
    manager=None
    try:
        configure()
        runtime=metadata()
        if runtime['tf32_enabled'] or not runtime['deterministic_algorithms']:
            raise RuntimeError('Numerical runtime contract inactive')
        tasks=json.loads((Path(__file__).resolve().parents[1]/'assets/eval_tasks.json').read_text())
        if isinstance(tasks,dict):tasks=tasks['tasks']
        manager=ManagedPolicy(adapter,{**details['identity'],**runtime},{t['task']:t['horizon'] for t in tasks},seed_scope=seeded)
        print(json.dumps({'model_ready':manager.metadata}),flush=True)
        asyncio.run(serve(manager,'127.0.0.1',a.port))
    finally:
        if manager is not None:manager.close()
        else:adapter.close()


if __name__=='__main__':main()
