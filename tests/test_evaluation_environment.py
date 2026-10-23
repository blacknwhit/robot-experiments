"""Launch-wiring regressions; real CPU children inspect inherited environments.

Policy loading, WebSocket and MuJoCo are mocked, so these are not GPU/EGL smoke.
"""
import concurrent.futures
import json
import os
import errno
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate

REAL_RUN = subprocess.run
REAL_POPEN = subprocess.Popen
ENV_KEYS = ('CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID','MUJOCO_GL',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')


def child_environment(env):
    code='import json,os; print(json.dumps({k:os.environ.get(k) for k in '+repr(ENV_KEYS)+'}))'
    result=REAL_RUN([sys.executable,'-c',code],env=env,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


class EvaluationEnvironmentTests(unittest.TestCase):
    def test_live_listener_is_still_rejected(self):
        with socket.socket() as server:
            option=getattr(socket,'SO_EXCLUSIVEADDRUSE',socket.SO_REUSEADDR)
            server.setsockopt(socket.SOL_SOCKET,option,1)
            server.bind(('127.0.0.1',0));server.listen(1)
            with self.assertRaises(OSError):evaluate.check_port_available(server.getsockname()[1])

    @unittest.skipUnless(sys.platform.startswith('linux'),'POSIX TCP TIME_WAIT regression')
    def test_completed_server_can_restart_without_waiting_for_time_wait(self):
        port=0
        for _ in range(2):
            with socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                server.settimeout(2);server.bind(('127.0.0.1',port));server.listen(1)
                port=server.getsockname()[1]
                with socket.create_connection(('127.0.0.1',port),timeout=2) as client:
                    accepted,_=server.accept()
                    with accepted:
                        accepted.settimeout(2);accepted.shutdown(socket.SHUT_WR)
                        self.assertEqual(client.recv(1),b'')
                        client.shutdown(socket.SHUT_WR)
                        self.assertEqual(accepted.recv(1),b'')
            with socket.socket() as old_probe:
                with self.assertRaises(OSError) as error:old_probe.bind(('127.0.0.1',port))
                self.assertEqual(error.exception.errno,errno.EADDRINUSE)
            evaluate.check_port_available(port)

    def test_simulator_preload_is_explicit_and_does_not_mutate_policy(self):
        with patch.dict(os.environ,{'LD_PRELOAD':'/policy/nccl.so','CUDA_VISIBLE_DEVICES':'3'}):
            clean=evaluate.simulator_environment('embedding3d_pos_a','2')
            pinned=evaluate.simulator_environment('embedding3d_pos_a','2',Path('/sim/libffi.so'))
            self.assertNotIn('LD_PRELOAD',clean)
            self.assertEqual(pinned['LD_PRELOAD'],str(Path('/sim/libffi.so')))
            self.assertEqual(pinned['PYTHONFAULTHANDLER'],'1')
            self.assertNotIn('CUDA_VISIBLE_DEVICES',pinned)
            self.assertEqual(pinned['MUJOCO_EGL_DEVICE_ID'],'2')
            self.assertNotIn('server_compat',pinned['PYTHONPATH'])
            self.assertEqual(os.environ['LD_PRELOAD'],'/policy/nccl.so')


    def test_explicit_mapping_and_invalid_assignments(self):
        self.assertEqual(evaluate.gpu_assignments('2, 5'),(['2','5'],['2','5']))
        self.assertEqual(evaluate.gpu_assignments('GPU-a,GPU-b','3,1'),(['GPU-a','GPU-b'],['3','1']))
        for cuda,egl in [('0,1','0'),('0,1','1,1'),('0,1','0,00'),('0,1','-1,0'),
                         ('0,','0,1'),('0,0','0,1'),('GPU-a',None)]:
            with self.subTest(cuda=cuda,egl=egl),self.assertRaises(ValueError):
                evaluate.gpu_assignments(cuda,egl)

    def check_parallel_launches(self,explicit_egl):
        servers={};simulators={};sim_commands=[]
        identity={'variant':'baseline','checkpoint_sha256':'weights','config_sha256':'config'}
        parent={'CUDA_VISIBLE_DEVICES':'7,3','MUJOCO_EGL_DEVICE_ID':'6',
                'OMP_NUM_THREADS':'88','MKL_NUM_THREADS':'99','OPENBLAS_NUM_THREADS':'32'}

        def launch_server(command,**kwargs):
            if command[1]=='-c':return REAL_POPEN(command,**kwargs)
            observed=child_environment(kwargs['env'])
            servers[observed['CUDA_VISIBLE_DEVICES']]=observed
            proc=MagicMock();proc.poll.return_value=None
            return proc

        def launch_simulator(command,**kwargs):
            observed=child_environment(kwargs['env'])
            simulators[observed['MUJOCO_EGL_DEVICE_ID']]=observed
            sim_commands.append(command)
            return subprocess.CompletedProcess(command,0)

        client=MagicMock();client.get_server_metadata.return_value=identity
        module=SimpleNamespace(WebsocketClientPolicy=MagicMock(return_value=client))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,parent):
            a=SimpleNamespace(variant='baseline',output=Path(tmp)/'eval',smoke=True,
                              home=Path(tmp),config=Path(tmp)/'cfg.yaml',checkpoint=Path(tmp)/'weights',
                              sim_python=Path(sys.executable),part=1,port_base=10090)
            with patch('evaluate.subprocess.Popen',side_effect=launch_server), \
                 patch('evaluate.subprocess.run',side_effect=launch_simulator), \
                 patch('evaluate.socket.socket'),patch('evaluate.socket.create_connection'), \
                 patch.dict(sys.modules,{'deployment.model_server.tools.websocket_policy_client':module}), \
                 patch('evaluate.validate_result',return_value={'successes':[False,False]}):
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    jobs=[pool.submit(evaluate.execute_slot,a,evaluate.TASKS[i],str(i),i,identity,
                                      None if explicit_egl is None else explicit_egl[i]) for i in range(2)]
                    for job in jobs:job.result()
            self.assertEqual({k:os.environ[k] for k in parent},parent)
        self.assertEqual(set(servers),{'0','1'})
        self.assertEqual(set(simulators),set(explicit_egl or ['0','1']))
        for env in [*servers.values(),*simulators.values()]:
            self.assertEqual([env[k] for k in ENV_KEYS[-3:]],['4','4','1'])
        for env in simulators.values():
            self.assertIsNone(env['CUDA_VISIBLE_DEVICES'])
            self.assertEqual(env['MUJOCO_GL'],'egl')
        for command in sim_commands:
            for flag,value in [('--n-action-steps','5'),('--n-envs','1'),('--seed','7'),
                               ('--strict-determinism','1'),('--renderer-backend','egl')]:
                self.assertEqual(command[command.index(flag)+1],value)

    def test_two_default_slots_bind_distinct_renderers_and_cap_cpu_children(self):
        self.check_parallel_launches(None)

    def test_two_remapped_slots_keep_cuda_and_egl_namespaces_separate(self):
        self.check_parallel_launches(['2','5'])


if __name__=='__main__':unittest.main()
