"""Exercise task isolation, real child environments, and concurrent smoke gates."""
import json,os,subprocess,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate
from common import environment
REAL_RUN=subprocess.run;REAL_POPEN=subprocess.Popen

class SharedGpuTests(unittest.TestCase):
    def test_task_coverage_and_resource_caps(self):
        for part in (1,13):
            tasks=evaluate.tasks_for_part(part)
            for gpus in (['0'],['0','1']):
                for per_gpu in (2,4):
                    plan=evaluate.task_slot_plan(tasks,gpus,gpus,per_gpu)
                    jobs=[i for w in plan for i,_ in w['jobs']]
                    self.assertEqual(sorted(jobs),list(range(len(tasks))))
                    self.assertEqual(len(jobs),len(set(jobs)))
                    for gpu in gpus:self.assertLessEqual(sum(w['gpu']==gpu for w in plan),per_gpu)
        with self.assertRaises(ValueError):evaluate.task_slot_plan(evaluate.TASKS[:4],['0'],['0'])

    def test_smoke_exercises_shared_gpu_and_rejects_insufficient_tasks(self):
        plan=evaluate.task_slot_plan(evaluate.TASKS[:4],['0','1'],['2','3'],2,True)
        self.assertEqual([(w['gpu'],w['egl']) for w in plan],[('0','2'),('0','2')])
        self.assertEqual([w['jobs'][0][0] for w in plan],[0,1])
        with self.assertRaises(ValueError):evaluate.task_slot_plan(evaluate.TASKS[:2],['0'],['0'],4,True)

    def test_real_worker_pool_never_exceeds_two_and_runs_each_task_once(self):
        plan=evaluate.task_slot_plan(evaluate.TASKS[:4],['0'],['0'],2)
        barrier=threading.Barrier(2,timeout=5);lock=threading.Lock();seen=[];active=0;peak=0
        def execute(a,task,gpu,slot,identity,egl):
            nonlocal active,peak
            with lock:active+=1;peak=max(peak,active);seen.append(slot)
            if slot<2:barrier.wait()
            time.sleep(.01)
            with lock:active-=1
            return slot
        with patch('evaluate.execute_slot',side_effect=execute):
            result=evaluate.execute_plan(SimpleNamespace(smoke=False),plan,{})
        self.assertEqual(sorted(seen),[0,1,2,3]);self.assertEqual(sorted(result),[0,1,2,3]);self.assertEqual(peak,2)

    def test_execution_identity_distinguishes_configuration_not_gpu_count(self):
        defaults=SimpleNamespace(tasks_per_gpu=1,policy_cpu_threads=4,sim_cpu_threads=4)
        self.assertEqual(evaluate.execution_identity(defaults),{})
        shared=SimpleNamespace(tasks_per_gpu=2,policy_cpu_threads=4,sim_cpu_threads=1,part=1,gpu_ids='0,1')
        identity=evaluate.execution_identity(shared)
        self.assertNotEqual(identity,{})
        shared.part=13;shared.gpu_ids='0'
        self.assertEqual(evaluate.execution_identity(shared),identity)
        shared.sim_cpu_threads=4
        self.assertNotEqual(evaluate.execution_identity(shared),identity)

    def test_shared_smoke_does_not_reuse_partial_results_or_leave_old_gate_active(self):
        with tempfile.TemporaryDirectory() as temp:
            a=SimpleNamespace(output=Path(temp));gate=a.output/'SMOKE_ACCEPTED.json';identity={'execution':{'tasks_per_gpu':2}}
            partial=a.output/'smoke'/'old-attempt';partial.mkdir(parents=True);(partial/'CloseBlenderLid.json').write_text('{}')
            evaluate.prepare_shared_smoke(a,gate,identity);first=a.smoke_attempt
            gate.write_text(json.dumps({'identity':identity,'application_smoke_passed':True}))
            evaluate.prepare_shared_smoke(a,gate,identity)
            self.assertNotEqual(first,a.smoke_attempt);self.assertFalse(gate.exists())
            self.assertTrue((a.output/'smoke'/a.smoke_attempt/'previous_gate.json').exists())
            self.assertTrue((partial/'CloseBlenderLid.json').exists())
            gate.write_text(json.dumps({'identity':{'foreign':True}}))
            with self.assertRaises(ValueError):evaluate.prepare_shared_smoke(a,gate,identity)
            self.assertTrue(gate.exists())

    def test_smoke_requires_real_common_rollout_interval(self):
        row=lambda a,b:{'rollout_timing':{'started_at':a,'finished_at':b}}
        self.assertEqual(evaluate.smoke_overlap([row(10,30),row(20,40)]),10)
        for results in ([row(10,20),row(20,40)],[row(10,20),{}],[row(10,float('inf')),row(12,20)]):
            with self.assertRaises(ValueError):evaluate.smoke_overlap(results)

    def test_shared_slots_use_distinct_ports_and_role_specific_real_child_envs(self):
        keys=('CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')
        observed={'policy':{},'sim':{}};commands=[]
        def inspect_env(env):
            command=[sys.executable,'-c','import json,os;print(json.dumps({k:os.environ.get(k) for k in '+repr(keys)+'}))']
            return json.loads(REAL_RUN(command,env=env,capture_output=True,text=True,check=True).stdout)
        def policy(command,**kwargs):
            if command[1]=='-c':return REAL_POPEN(command,**kwargs)
            port=command[command.index('--port')+1];observed['policy'][port]=inspect_env(kwargs['env'])
            p=MagicMock();p.poll.return_value=None;return p
        def simulator(command,**kwargs):
            port=command[command.index('--port')+1];observed['sim'][port]=inspect_env(kwargs['env']);commands.append(command)
            return subprocess.CompletedProcess(command,0)
        identity={'variant':'baseline','checkpoint_sha256':'cp','config_sha256':'cfg'}
        client=MagicMock();client.get_server_metadata.return_value=identity
        module=SimpleNamespace(WebsocketClientPolicy=MagicMock(return_value=client))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'OMP_NUM_THREADS':'99','MKL_NUM_THREADS':'88'}):
            a=SimpleNamespace(variant='baseline',output=Path(tmp),smoke=True,home=Path(tmp),config=Path(tmp)/'cfg',
                checkpoint=Path(tmp)/'cp',sim_python=Path(sys.executable),part=1,port_base=12300,tasks_per_gpu=2,policy_cpu_threads=4,sim_cpu_threads=1)
            plan=evaluate.task_slot_plan(evaluate.TASKS[:4],['0'],['2'],2,True)
            with patch('evaluate.subprocess.Popen',side_effect=policy),patch('evaluate.subprocess.run',side_effect=simulator), \
                 patch('evaluate.socket.socket'),patch('evaluate.socket.create_connection'), \
                 patch.dict(sys.modules,{'deployment.model_server.tools.websocket_policy_client':module}), \
                 patch('evaluate.validate_result',return_value={}):evaluate.execute_plan(a,plan,identity)
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'99')
        self.assertEqual(set(observed['policy']),{'12300','12301'});self.assertEqual(set(observed['sim']),{'12300','12301'})
        for env in observed['policy'].values():self.assertEqual([env[k] for k in keys],['0',os.environ.get('MUJOCO_EGL_DEVICE_ID'),'4','4','1'])
        for env in observed['sim'].values():self.assertEqual([env[k] for k in keys],[None,'2','1','1','1'])
        for cmd in commands:
            for flag,value in [('--n-envs','1'),('--seed','7'),('--n-action-steps','5')]:self.assertEqual(cmd[cmd.index(flag)+1],value)

    def test_failed_server_aborts_shared_barrier_and_cleans_other_server(self):
        identity={'variant':'baseline','checkpoint_sha256':'cp','config_sha256':'cfg'}
        client=MagicMock();client.get_server_metadata.return_value=identity
        module=SimpleNamespace(WebsocketClientPolicy=MagicMock(return_value=client))
        servers=[]
        def start(command,**kwargs):
            server=MagicMock();port=command[command.index('--port')+1]
            server.poll.return_value=1 if port=='12300' else None
            servers.append(server);return server
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(variant='baseline',output=Path(tmp),smoke=True,home=Path(tmp),config=Path(tmp)/'cfg',
                checkpoint=Path(tmp)/'cp',sim_python=Path(sys.executable),part=1,port_base=12300,tasks_per_gpu=2,policy_cpu_threads=4,sim_cpu_threads=1)
            plan=evaluate.task_slot_plan(evaluate.TASKS[:4],['0'],['0'],2,True)
            with patch('evaluate.subprocess.Popen',side_effect=start),patch('evaluate.subprocess.run') as sim, \
                 patch('evaluate.socket.socket'),patch('evaluate.socket.create_connection'), \
                 patch.dict(sys.modules,{'deployment.model_server.tools.websocket_policy_client':module}):
                with self.assertRaisesRegex(RuntimeError,'Server exited'):evaluate.execute_plan(a,plan,identity)
                sim.assert_not_called()
            self.assertTrue(a.smoke_server_barrier.broken)
        self.assertEqual(len(servers),2)
        for server in servers:server.terminate.assert_called_once();server.wait.assert_called_once()

    def test_thread_values_reject_invalid_input(self):
        for value in (0,-1,True,'4'):
            with self.assertRaises(ValueError):environment('baseline',cpu_threads=value)
if __name__=='__main__':unittest.main()
