"""Feed an old test_cpu.py reference checkpoint to the real run.main resume path."""
import argparse
from contextlib import ExitStack
import dataclasses
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path, help="Old test_cpu.py --mode reference output")
    p.add_argument("--output", required=True, type=Path, help="New sibling directory for this test")
    args = p.parse_args()
    old = args.reference.resolve()
    dest = args.output.resolve()
    dest.mkdir(exist_ok=False)
    source = old / "checkpoints/step_000002"
    scope = old.parent
    dest.relative_to(scope)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    from environment import setup
    cfg = json.loads((EXP/"recipe_8gpu64_70k.json").read_text())
    cfg.update(workspace_root=str(scope), run_root=str(dest), mode="smoke", deterministic_gpu=False,
               tokenizer=str(dest/"tokenizer"), norm_assets=str(dest/"norm"), base_checkpoint=str(dest/"absent-base"),
               source_commit="legacy", source_files={"old.py":"old"}, reader={"num_workers":4,
               **{k:str(dest) for k in ("data_root_dir","official_registry_file","v3_arrow_cache_dir")}},
               resource_manifest={str(dest/"absent-base/weights"):{"bytes":1,"sha256":"unused"}})
    setup(cfg,cpu=True)
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import checkpoints
    import run
    from compare_runs import signature
    from contract import load
    from openpi.shared import array_typing,normalize
    from openpi.training import sharding,utils

    # Match the published legacy CPU model and its deterministic indexed minibatches.
    class Tiny(nnx.Module):
        def __init__(self):
            r=nnx.Rngs(42)
            self.PaliGemma=nnx.Dict(img=nnx.Linear(2048,512,rngs=r),llm=nnx.Linear(512,32,rngs=r))
            self.action=nnx.Linear(32,1,rngs=r)
        def compute_loss(self,rng,observation,actions,train=False):
            x=nnx.swish(self.PaliGemma.img(observation))
            x=nnx.swish(self.PaliGemma.llm(x))
            return jnp.square(self.action(x)+.01*jax.random.normal(rng,actions.shape)-actions)

    def init(conf,ir,mesh,resume=False):
        assert resume, "Must restore, never initialize from base"
        model=Tiny()
        params=nnx.state(model)
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(.001,b1=.9,b2=.95,eps=1e-8,weight_decay=1e-10))
        state=utils.TrainState(step=0,params=params,model_def=nnx.graphdef(model),tx=tx,opt_state=tx.init(params),
                               ema_decay=.999,ema_params=params)
        shape=jax.eval_shape(lambda:state)
        return shape,sharding.fsdp_sharding(shape,mesh)

    ranges=[]
    def batches(windows,proc,cfg,cursor,stop,*,timing):
        ranges.append((cursor,stop))
        for index in range(cursor,stop,64):
            g=np.random.default_rng(index//64)
            yield (g.normal(size=(64,2048)).astype(np.float32),g.normal(size=(64,1)).astype(np.float32))

    def cpu_topology(jax,cfg):
        assert len(jax.devices())==8 and all(d.platform=="cpu" for d in jax.devices())

    conf=SimpleNamespace(trainable_filter=nnx.Param,lr_schedule=SimpleNamespace(create=lambda:lambda step:jnp.float32(.001)))
    normalize.save(Path(cfg["norm_assets"]),{"state":normalize.NormStats(mean=np.zeros(1),std=np.ones(1))})
    config=dest/"input.json"
    config.write_text(json.dumps(cfg))
    legacy_formal=dest/"formal-input.json"
    legacy_formal.write_text(json.dumps(cfg|{"mode":"formal","gpu_acceptance":{
        "passed":True,"physical_gpu_count":8,"recipe_identity":"previous-release"}}))
    assert load(legacy_formal)["mode"]=="formal"
    output=dest/"training"
    golden=json.loads((old/"REFERENCE.json").read_text())
    source_marker=(source/"COMPLETE.json").read_bytes()
    with ExitStack() as stack:
        for name,value in (("run.require_gpu_topology",cpu_topology),("engine.make_config",lambda cfg:conf),
                           ("official_init.init_train_state",init),("data.DemoStream",lambda cfg:None),
                           ("data.Processor",lambda cfg:None),("input_pipeline.Windows",lambda ds:None),
                           ("input_pipeline.batches",batches)):
            stack.enter_context(patch(name,value))
        def launch(target, *cli):
            argv=["run.py","--config",str(config),"--run-root",str(output),"--stop-after",str(target),*map(str,cli)]
            with patch.object(sys,"argv",argv):
                run.main()
        # Simulate a preemption after a source has been claimed, before the first local save.
        with patch("input_pipeline.batches",side_effect=InterruptedError("test preemption")):
            try:launch(3,"--resume-from",source,"--loader-workers","8")
            except InterruptedError:pass
            else:raise AssertionError("Expected simulated preemption")
        assert not (output/"checkpoints").exists()
        launch(3,"--loader-workers","8")
        mesh=sharding.make_mesh(8)
        shape,ss=init(conf,None,mesh,resume=True)
        restored,rng,cursor=checkpoints.restore(output/"checkpoints/step_000003",shape,ss,jax.random.key(0),cfg,mesh)
        assert signature(restored,rng,cursor)==golden
        cfg["reader"]["num_workers"]=8
        config.write_text(json.dumps(cfg))
        launch(4,"--resume-from",source)
        launch(5)
        assert ranges==[(128,192),(192,256),(256,320)]
        rows=[json.loads(x) for x in (output/"attempts.jsonl").read_text().splitlines()]
        assert [x["start_step"] for x in rows]==[2,2,3,4]
        assert rows[0]["restored_from"]==str(source)
        assert rows[1]["restored_from"]==str(source) and rows[1]["loader_workers"]==8
        assert rows[2]["restored_from"]==str(output/"checkpoints/step_000003")
        try:checkpoints.select_resume(output,cfg,source,{"identity":"another-existing-run"})
        except ValueError:pass
        else:raise AssertionError("Conflicting destination accepted")
        bad_params=jax.tree.map(lambda a:jax.ShapeDtypeStruct((2,),a.dtype) if a.shape==(1,) else a,shape.params)
        with array_typing.disable_typechecking():
            incompatible=dataclasses.replace(shape,params=bad_params)
        try:checkpoints.restore(source,incompatible,ss,jax.random.key(0),cfg,mesh)
        except ValueError as exc:assert 'Incompatible restored shape or dtype' in str(exc)
        else:raise AssertionError("Mismatched parameter shape accepted")
        # A saved sample cursor can have an offset when batch size changed earlier.
        offset=dest/"offset"
        point,_=checkpoints.save(offset,restored,rng,257,cfg)
        output=dest/"offset-training"
        launch(4,"--resume-from",point)
        assert ranges[-1]==(257,321)
        assert json.loads((output/"checkpoints/step_000004/COMPLETE.json").read_text())["next_sample"]==321
        for bad in (old/"checkpoints/step_999999",output/"exports/step_000004",dest/"missing"):
            try:checkpoints.select_resume(dest/"empty",cfg,bad,None)
            except (ValueError,FileNotFoundError):pass
            else:raise AssertionError(f"Invalid source accepted: {bad}")
        # Corrupt a file only in this test's own copy.
        file=next((point/"state").rglob("*.zarray"),None)
        if file is None:file=next(f for f in (point/"state").rglob("*") if f.is_file())
        with file.open("ab") as f:f.write(b"!")
        try:checkpoints.verify_step(point,cfg)
        except ValueError:pass
        else:raise AssertionError("Corrupted checkpoint accepted")
    assert (source/"COMPLETE.json").read_bytes()==source_marker
    checkpoints.verify_step(source,cfg)
    print("LEGACY_RESUME_ACCEPTED",json.dumps({"old_step":2,"exact_step3_signature":golden,"starts":[2,2,3,4],
             "cursor_ranges":ranges,"invalid_and_corrupt_rejected":True,"virtual_cpu_devices":8}),flush=True)


if __name__=="__main__":
    main()
