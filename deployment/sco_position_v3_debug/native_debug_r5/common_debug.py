from pathlib import Path
import hashlib,json,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    for directory,filename,expected in [(ROOT,'release_manifest.json','65c003c6766692a299eed7f58d075bbc7060e1ed8c0338f8bfec4c198a3c0dbb'),
        (ROOT/'eval_repair_r4_20261022','manifest.json','7d8ffef7dc9901625746c75a09755e80afab241b9138ea24135f85b7228e2b96')]:
        assert digest(directory/filename)==expected
        for name,sha in json.loads((directory/filename).read_text())['files'].items():
            p=directory/name;p.resolve().relative_to(ROOT);assert digest(p)==sha,name
    for name,sha in json.loads((HERE/'manifest.json').read_text())['files'].items():
        p=HERE/name;p.resolve().relative_to(HERE);assert digest(p)==sha,name
def command(out):
    return [sys.executable,str(HERE/'run_eval.py'),'--variant','embedding3d_pos_a',
        '--home',str(ROOT/'home'),'--config',str(ROOT/'configs/embedding3d_pos_a_short_1500.yaml'),
        '--checkpoint',str(ROOT/'home/engineering_runs/embedding3d_pos_a_short_r2/checkpoints/steps_1500_model.safetensors'),
        '--sim-python',str(HERE/'sim_debug.sh'),'--sim-ld-preload','/lib/x86_64-linux-gnu/libffi.so.7',
        '--output',str(out),'--gpu-ids','0','--port-base','11390','--smoke']
def native_summary(out):
    p=Path(out)/'runtime_audit/native/native.json'
    if not p.exists():return None
    d=json.loads(p.read_text())
    return {k:d.get(k) for k in ('signal','exit_code','timeout_triggered','inferior_pid','target_environment')}
