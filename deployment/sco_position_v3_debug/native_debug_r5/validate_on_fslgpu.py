from pathlib import Path
import json,socket,subprocess,time
from common_debug import HERE,verify,command,native_summary
from gdb_runner import prerequisites
verify();prerequisites()
out=HERE/('fslgpu_'+str(time.time_ns()));out.mkdir()
with (out/'driver.log').open('x') as log:
    proc=subprocess.run(command(out),stdout=log,stderr=subprocess.STDOUT)
native=native_summary(out)
proof=dict(time=time.time(),hostname=socket.gethostname(),output=str(out),returncode=proc.returncode,native=native,
           diagnostic_harness_passed=bool(native and (native.get('signal') or native.get('exit_code')==0)),
           rendering_bug_fixed=False,training_started=False)
if proc.returncode==0:proof['smoke']=json.loads((out/'SMOKE_ACCEPTED.json').read_text())
(HERE/'FSLGPU_DIAGNOSTIC_HARNESS.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
assert proof['diagnostic_harness_passed']
