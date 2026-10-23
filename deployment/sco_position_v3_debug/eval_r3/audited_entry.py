"""Instrumentation only: execute the unmodified public server/simulator entry."""
from pathlib import Path
import json,os,runpy,sys,time,traceback
from gpu_identity import cuda_identity,current_egl_uuid

A=Path(os.environ['EVAL_AUDIT_ATTEMPT'])
role=sys.argv.pop(1)
key=os.environ.get('CUDA_VISIBLE_DEVICES','') if role=='policy' else os.environ['MUJOCO_EGL_DEVICE_ID']
proof=dict(role=role,pid=os.getpid(),started=time.time(),environment={k:os.environ.get(k) for k in
    ('CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_PRELOAD','PYTHONFAULTHANDLER')})
output=A/f'{role}_{key}.json'


def save():
    output.write_text(json.dumps(proof,indent=2))


try:
    if role=='policy':
        proof['cuda']=cuda_identity();assert len(proof['cuda'])==1
        save();sys.argv=sys.argv[1:];sys.path.insert(0,str(Path(sys.argv[0]).parent))
        runpy.run_path(sys.argv[0],run_name='__main__')
    else:
        import numpy as np
        import robocasa,robosuite
        assert robocasa.__version__=='1.0.0' and robosuite.__version__=='1.5.2'
        from robosuite.renderers.context import egl_context
        make_current=egl_context.EGLGLContext.make_current
        def checked_make_current(self):
            result=make_current(self)
            if 'actual_egl_uuid' not in proof:
                from OpenGL.GL import glGetString,GL_RENDERER,GL_VERSION
                proof.update(actual_egl_uuid=current_egl_uuid(),gl_renderer=glGetString(GL_RENDERER).decode(),gl_version=glGetString(GL_VERSION).decode())
                assert 'NVIDIA' in proof['gl_renderer'] or 'GeForce' in proof['gl_renderer']
                save()
            return result
        egl_context.EGLGLContext.make_current=checked_make_current
        from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
        for method in ('step','reset'):
            original=getattr(RoboCasaGymEnv,method)
            def timed(self,*args,_original=original,_method=method,**kwargs):
                start=time.perf_counter();result=_original(self,*args,**kwargs)
                proof[_method+'_seconds']=proof.get(_method+'_seconds',0)+time.perf_counter()-start
                proof[_method+'_calls']=proof.get(_method+'_calls',0)+1
                if proof[_method+'_calls']%100==0:save()
                return result
            setattr(RoboCasaGymEnv,method,timed)
        original_obs=RoboCasaGymEnv.get_observation
        def check_obs(self,*args,**kwargs):
            value=original_obs(self,*args,**kwargs)
            if 'image_shapes' not in proof:
                images=[np.asarray(v) for k,v in value.items() if k.startswith('video.')]
                assert len(images)==3 and all(np.isfinite(v).all() and float(v.std())>0 for v in images)
                proof['image_shapes']=[list(v.shape) for v in images];save()
            return value
        RoboCasaGymEnv.get_observation=check_obs
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        predict=WebsocketClientPolicy.predict_action
        def measured_predict(self,*args,**kwargs):
            t=time.perf_counter();result=predict(self,*args,**kwargs)
            proof['rpc_seconds']=proof.get('rpc_seconds',0)+time.perf_counter()-t
            proof['rpc_calls']=proof.get('rpc_calls',0)+1
            return result
        WebsocketClientPolicy.predict_action=measured_predict
        assert sys.argv[1]=='-m';module=sys.argv[2];sys.argv=[module,*sys.argv[3:]]
        runpy.run_module(module,run_name='__main__')
        proof['completed']=True
except BaseException:
    proof['exception']=traceback.format_exc();raise
finally:
    proof['ended']=time.time();save()
