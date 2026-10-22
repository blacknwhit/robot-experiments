"""Extract an isolated debugger under RobotVla; do not install system packages."""
from pathlib import Path
import hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent
HERE.resolve().relative_to(Path('/data/xiangyuyue/ULLM-zf/RobotVla'))
packages=HERE/'tools/packages';prefix=HERE/'tools/root'
packages.mkdir(parents=True,exist_ok=True);prefix.mkdir(parents=True,exist_ok=True)
names=['gdb','libbabeltrace1','libpython3.8','libpython3.8-stdlib','libpython3.8-minimal',
       'libmpfr6','libreadline8','libncursesw6','libtinfo6','libdw1']
subprocess.run(['apt-get','download',*names],cwd=packages,check=True)
receipts=[]
for p in sorted(packages.glob('*.deb')):
    unpack=subprocess.Popen(['dpkg-deb','--fsys-tarfile',str(p)],stdout=subprocess.PIPE)
    subprocess.run(['tar','-xf','-','--no-same-owner','--no-same-permissions','-C',str(prefix)],
                   stdin=unpack.stdout,check=True)
    unpack.stdout.close();assert unpack.wait()==0
    receipts.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
env=dict(os.environ,LD_LIBRARY_PATH=str(prefix/'usr/lib/x86_64-linux-gnu')+':'+str(prefix/'lib/x86_64-linux-gnu'),
         PYTHONHOME=str(prefix/'usr'),PYTHONPATH='')
env.pop('LD_PRELOAD',None)
gdb=prefix/'usr/bin/gdb'
version=subprocess.run([str(gdb),'--version'],env=env,capture_output=True,text=True)
proof=dict(time=time.time(),packages=receipts,version_stdout=version.stdout,
           version_stderr=version.stderr,returncode=version.returncode)
(HERE/'gdb_prepare.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='packages'},indent=2));version.check_returncode()
