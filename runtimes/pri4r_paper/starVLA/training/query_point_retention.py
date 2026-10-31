"""Retain full milestones/latest three; delete only owned complete saves."""
from pathlib import Path
import importlib.util,json,shutil
def retained_steps(steps):
    return set(sorted(set(steps))[-3:]) | ({1,1000,30000,50000,70000} & set(steps))
def complete_and_prune(output,step,world_size,accumulation):
    assert (world_size,accumulation)==(2,1)
    source=Path(__file__).resolve().parents[2]
    meta=json.loads((source/'examples/pi05_robocasa365/trajectory_v2/run.json').read_text())
    output=Path(output).resolve();allowed=source/'outputs'/meta['run_id']
    assert output==allowed and source==Path(meta['target_root'])
    p=source/'examples/pi05_robocasa365/trajectory_v2/recovery.py'
    spec=importlib.util.spec_from_file_location('trajectory_recovery_retention',p)
    recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
    root=output/'checkpoints';recovery.validate_checkpoint(root/f'checkpoint-{step}',output)
    marker=output/'query_point_checkpoints.json'
    record=json.loads(marker.read_text()) if marker.exists() else dict(owner=str(output),complete_steps=[])
    assert record['owner']==str(output)
    steps=sorted(set(record['complete_steps']+[step]));keep=retained_steps(steps)
    record['complete_steps']=steps
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(marker)
    for old in steps:
        if old in keep:continue
        for path in [root/f'checkpoint-{old}',root/f'steps_{old}_model.safetensors']:
            assert not path.is_symlink() and path.resolve().parent==root.resolve()
            if path.is_dir():shutil.rmtree(path)
            elif path.is_file():path.unlink()
    record['complete_steps']=sorted(keep)
    tmp.write_text(json.dumps(record,indent=2));tmp.replace(marker)
