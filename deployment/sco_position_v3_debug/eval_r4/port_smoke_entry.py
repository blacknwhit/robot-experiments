"""Real evaluation with a freshly closed TCP connection before the port probe."""
from pathlib import Path
import errno,importlib.util,json,os,socket,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'repo/scripts'))
spec=importlib.util.spec_from_file_location('tested_evaluator',HERE/'evaluate.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
output=Path(sys.argv[sys.argv.index('--output')+1])
(output/'runtime_audit').mkdir(parents=True,exist_ok=True)
os.environ['EVAL_AUDIT_ATTEMPT']=str(output/'runtime_audit')
original=evaluation.check_port_available
def close_then_check(port):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.settimeout(2);server.bind(('127.0.0.1',port));server.listen(1)
        with socket.create_connection(('127.0.0.1',port),timeout=2) as client:
            accepted,_=server.accept()
            with accepted:
                accepted.settimeout(2);accepted.shutdown(socket.SHUT_WR)
                assert client.recv(1)==b''
                client.shutdown(socket.SHUT_WR);assert accepted.recv(1)==b''
    error=None
    with socket.socket() as old:
        try:old.bind(('127.0.0.1',port))
        except OSError as exc:error=exc.errno
    assert error==errno.EADDRINUSE
    original(port)
    (output/'PORT_REUSE_ACCEPTED.json').write_text(json.dumps(dict(time=time.time(),port=port,
        old_probe_errno=error,new_probe_passed=True,real_tcp_time_wait=True),indent=2))
evaluation.check_port_available=close_then_check
evaluation.main()
