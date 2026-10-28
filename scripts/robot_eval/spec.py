"""Declarative local model loading, separate from the sealed benchmark code."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from . import API_VERSION

FORMAT = 'robot_eval_model_spec_v1'
ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = ('LD_PRELOAD','LD_LIBRARY_PATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'TOKENIZERS_PARALLELISM','HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE',
            'XDG_CACHE_HOME','OPENPI_DATA_HOME','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','TMPDIR')
FIXED_ENV = {'PYTHONHASHSEED':'7','CUBLAS_WORKSPACE_CONFIG':':4096:8',
             'NVIDIA_TF32_OVERRIDE':'0','TORCH_COMPILE_DISABLE':'1',
             'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
             'OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def inside(path, root):
    try:Path(path).resolve().relative_to(Path(root).resolve());return True
    except ValueError:return False


def _path(value, parent):
    if not isinstance(value,str) or not value:raise ValueError('Model paths must be nonempty strings')
    text=os.path.expandvars(os.path.expanduser(value))
    if '$' in text or re.search(r'%[A-Za-z_][A-Za-z_0-9]*%',text):
        raise ValueError('Unresolved model path variable: '+value)
    path=Path(text)
    return (path if path.is_absolute() else parent/path).resolve()


@dataclass
class ResolvedSpec:
    path: Path
    data: dict
    model_id: str
    factory: str
    config: Path
    checkpoint: Path
    policy_python: Path
    python_paths: tuple
    source_roots: tuple
    assets: dict

    def source_files(self):
        result={}
        for index,(root,patterns) in enumerate(self.source_roots):
            if not root.is_dir():raise ValueError('Missing model source root: '+str(root))
            found=0
            for pattern in patterns:
                if Path(pattern).is_absolute() or '..' in Path(pattern).parts:
                    raise ValueError('Source glob must stay under its declared root')
                for path in sorted(root.glob(pattern)):
                    if not path.is_file() or any(part in ('.git','__pycache__','.venv') for part in path.relative_to(root).parts):continue
                    if not inside(path,root):raise ValueError('Source symlink escapes declared root: '+str(path))
                    result[f'{index}/{path.relative_to(root).as_posix()}']=path.resolve();found+=1
            if not found:raise ValueError('Source patterns matched no files: '+str(root))
        return result

    def asset_files(self):
        result={}
        for name,root in self.assets.items():
            if root.is_file():result[name]=root
            elif root.is_dir():
                files=[p for p in sorted(root.rglob('*')) if p.is_file()]
                if not files:raise ValueError('Empty model asset directory: '+str(root))
                for path in files:
                    if not inside(path,root):raise ValueError('Asset symlink escapes declared directory')
                    result[name+'/'+path.relative_to(root).as_posix()]=path.resolve()
            else:raise ValueError('Missing model asset: '+str(root))
        return result

    def policy_environment(self, *, gpu=None, home=None):
        env=dict(os.environ)
        env.pop('PYTHONPATH',None)
        env.update(FIXED_ENV)
        if home is not None:env['OPENPI_DATA_HOME']=str(Path(home)/'cache/openpi')
        env.update(self.data.get('environment',{}))
        env.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts',*self.python_paths))),
                   CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
        return env

    def files(self):
        paths=[self.path,self.config,self.checkpoint,*self.source_files().values(),*self.asset_files().values()]
        return tuple(dict.fromkeys(paths))

    def details(self, environment_snapshot):
        for path in (self.config,self.checkpoint):
            if not path.is_file():raise ValueError('Missing model config/checkpoint: '+str(path))
        source={name:{'path':str(path),'sha256':file_sha(path)} for name,path in self.source_files().items()}
        assets={name:{'path':str(path),'sha256':file_sha(path)} for name,path in self.asset_files().items()}
        contract={key:self.data[key] for key in ('backend','backend_version','state_mode','rng_mode','max_concurrency','test_only')}
        identity={'variant':self.model_id,'model_adapter_api_version':API_VERSION,
                  'model_spec_sha256':file_sha(self.path),'checkpoint_sha256':file_sha(self.checkpoint),
                  'config_sha256':file_sha(self.config),'model_sources_sha256':digest(source),
                  'model_assets_sha256':digest(assets),'policy_environment_sha256':digest(environment_snapshot),
                  'inference_backend':contract.pop('backend'),'inference_backend_version':contract.pop('backend_version'),
                  **contract}
        details={'format':FORMAT,'spec_path':str(self.path),'spec':self.data,'factory':self.factory,
                 'config':str(self.config),'checkpoint':str(self.checkpoint),'policy_python':str(self.policy_python),
                 'python_paths':list(map(str,self.python_paths)),'sources':source,'assets':assets,
                 'environment':environment_snapshot,'identity':identity}
        identity['model_spec_fingerprint']=digest(details)
        return details


def load_spec(path, *, config=None, checkpoint=None):
    path=Path(path).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    allowed={'format','model_id','factory','config','checkpoint','policy_python','python_paths','source_roots',
             'assets','backend','backend_version','state_mode','rng_mode','max_concurrency','options','environment','test_only'}
    if not isinstance(data,dict) or set(data)-allowed:raise ValueError('Unknown model-spec fields')
    if data.get('format')!=FORMAT:raise ValueError('Unsupported model-spec format')
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',str(data.get('model_id',''))):raise ValueError('Invalid model_id')
    factory=data.get('factory','pi05')
    if factory!='pi05' and not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*',factory):
        raise ValueError('Factory must be pi05 or an importable module:function')
    data.setdefault('factory',factory)
    data.setdefault('backend','cuda-graph-pipeline' if factory=='pi05' else 'native')
    versions={'cuda-graph-pipeline':'cpu_gpu_pipeline_batch1_v1','native':'model_native_v1'}
    data.setdefault('backend_version',versions.get(data['backend']))
    for key in ('backend','backend_version'):
        if not isinstance(data[key],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',data[key]):
            raise ValueError('Explicit backend name/version required')
    fast=factory=='pi05' and data['backend']=='cuda-graph-pipeline'
    data.setdefault('state_mode','stateless');data.setdefault('rng_mode','explicit_per_request' if factory=='pi05' else 'managed_global')
    data.setdefault('max_concurrency',16 if fast else 1);data.setdefault('test_only',False)
    if data['state_mode'] not in ('stateless','per_episode') or data['rng_mode'] not in ('explicit_per_request','managed_global'):
        raise ValueError('Invalid state/RNG mode')
    if type(data['max_concurrency']) is not int or not 1<=data['max_concurrency']<=16:
        raise ValueError('max_concurrency must be1..16')
    if data['rng_mode']=='managed_global' and data['max_concurrency']!=1:
        raise ValueError('Managed RNG is serialized')
    if type(data['test_only']) is not bool:raise ValueError('test_only must be Boolean')
    data.setdefault('options',{});data.setdefault('environment',{})
    if not isinstance(data['options'],dict) or not isinstance(data['environment'],dict):raise ValueError('Options/environment must be mappings')
    if set(data['environment'])-set(ENV_KEYS) or any(not isinstance(v,str) for v in data['environment'].values()):
        raise ValueError('Unsupported policy environment override; GPU/seed/precision controls are fixed')
    parent=path.parent
    roots=[]
    for row in data.get('source_roots',[]):
        if not isinstance(row,dict) or set(row)-{'path','include'}:raise ValueError('Invalid source-root declaration')
        patterns=row.get('include',['**/*.py'])
        if not isinstance(patterns,list) or not patterns or any(not isinstance(p,str) or not p for p in patterns):
            raise ValueError('Source include patterns required')
        roots.append((_path(row['path'],parent),tuple(patterns)))
    if not roots:raise ValueError('Declare the actual model/adapter source roots')
    python_paths=tuple(_path(value,parent) for value in data.get('python_paths',[]))
    if not python_paths or any(not any(inside(p,r) for r,_ in roots) for p in python_paths):
        raise ValueError('Model python_paths must belong to declared source roots')
    assets=data.get('assets',{})
    if not isinstance(assets,dict) or any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',name) for name in assets):
        raise ValueError('Assets must be a name-to-path mapping')
    return ResolvedSpec(path,data,data['model_id'],factory,
                        Path(config).resolve() if config is not None else _path(data['config'],parent),
                        Path(checkpoint).resolve() if checkpoint is not None else _path(data['checkpoint'],parent),
                        _path(data.get('policy_python',sys.executable),parent),python_paths,tuple(roots),
                        {name:_path(value,parent) for name,value in assets.items()})


def environment_snapshot():
    """No GPU initialization or imports of the model's numerical libraries."""
    import importlib.metadata
    packages=sorted([str(d.metadata.get('Name','')).lower(),d.version] for d in importlib.metadata.distributions())
    keys=(*ENV_KEYS,'PYTHONPATH','PYTHONHASHSEED','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_COMPILE_DISABLE')
    return {'python':sys.version,'executable':str(Path(sys.executable).resolve()),'packages':packages,
            'environment':{key:os.environ.get(key) for key in keys}}


def probe_environment(spec, env):
    code='import json;from robot_eval.spec import environment_snapshot;print(json.dumps(environment_snapshot(),sort_keys=True))'
    result=subprocess.run([str(spec.policy_python),'-c',code],env=env,capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError('Model Python environment probe failed: '+result.stderr[-2000:])
    return json.loads(result.stdout)


def identity_summary(details):
    return dict(details['identity'])
