"""Run immutable r4 evaluation with the native-debug simulator wrapper."""
from pathlib import Path
import importlib.util,os,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'repo/scripts'))
spec=importlib.util.spec_from_file_location('r4_evaluate',ROOT/'eval_repair_r4_20261022/evaluate.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
output=Path(sys.argv[sys.argv.index('--output')+1]);output.resolve().relative_to(HERE)
(output/'runtime_audit').mkdir(parents=True,exist_ok=True)
os.environ['EVAL_AUDIT_ATTEMPT']=str(output/'runtime_audit')
module.main()
