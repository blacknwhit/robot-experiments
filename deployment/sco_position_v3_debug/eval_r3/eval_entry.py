"""Versioned eval-only overlay; immutable r2 model/trainer are retained."""
from pathlib import Path
import os,runpy,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
assert (ROOT/'release_manifest.json').is_file()
sys.path.insert(0,str(ROOT/'repo/scripts'))
output=Path(sys.argv[sys.argv.index('--output')+1])/'runtime_audit'
output.mkdir(parents=True,exist_ok=True)
os.environ['EVAL_AUDIT_ATTEMPT']=str(output)
runpy.run_path(str(HERE/'evaluate.py'),run_name='__main__')
