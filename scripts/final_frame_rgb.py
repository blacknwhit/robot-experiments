"""Skip only unread RGB renders; preserve the native simulator and RNG clocks."""
import math
import numpy as np


class CameraCache:
    def __init__(self, observable, guard):
        self.original=observable._sensor;self.guard=guard
        self.value=np.array(observable._current_observed_value,copy=True)
        self.rendered=0;self.reused=0
        self.__modality__=self.original.__modality__

    def __call__(self, obs_cache):
        if self.guard.skip:
            self.reused+=1
            return self.value
        self.value=self.original(obs_cache);self.rendered+=1
        return self.value


class Guard:
    def __init__(self, wrapper):
        from robosuite.utils.observables import NO_CORRUPTION,NO_FILTER,NO_DELAY
        env=wrapper.env;names=[name+'_image' for name in wrapper.camera_names]
        if not env.ignore_done or len(names)!=3:raise ValueError('RGB optimization requires the native three-camera, ignore_done environment')
        if any(env.camera_depths) or any(s is not None for s in env.camera_segmentations):raise ValueError('Depth/segmentation cannot reuse RGB')
        images={name for name,o in env._observables.items() if o._sensor.__modality__=='image'}
        if images!=set(names):raise ValueError('Unexpected image observables')
        self.skip=False;self.cameras=[];self.observables=tuple(env._observables[n] for n in names)
        self.steps=0;self.chunks=0
        for name in names:
            o=env._observables[name]
            if not o.is_enabled() or not o.is_active():raise ValueError('Camera disabled')
            if (o._corrupter is not NO_CORRUPTION or o._filter is not NO_FILTER or o._delayer is not NO_DELAY):raise ValueError('Nontrivial camera corruption/filter/delay')
            if not math.isclose(o._sampling_timestep,env.control_timestep,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Camera/control sampling rates differ')
            value=np.asarray(o._current_observed_value)
            if value.dtype!=np.uint8 or value.shape!=(wrapper.camera_heights,wrapper.camera_widths,3):raise ValueError('Camera cache is not native RGB')
            camera=CameraCache(o,self);o._sensor=camera;self.cameras.append(camera)

    def chunk_steps(self, wrapper):
        if list(wrapper.video_delta_indices)!=[0] or list(wrapper.state_delta_indices)!=[0]:raise ValueError('Historical observation windows need intermediate frames')
        remaining=wrapper.n_action_steps
        if wrapper.max_episode_steps is not None:remaining=min(remaining,wrapper.max_episode_steps-len(wrapper.reward))
        if remaining<1 or (wrapper.done and wrapper.done[-1]):raise ValueError('Completed environment must reset')
        return remaining


def install():
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
    old_chunk=MultiStepWrapper.step;old_control=RoboCasaGymEnv.step
    guards=[]
    retired={'control_steps':0,'chunks':0,'rgb_rendered':0,'rgb_reused':0}
    def chunk(self,actions):
        raw=self.env.unwrapped
        if not isinstance(raw,RoboCasaGymEnv):raise ValueError('Unsupported simulator wrapper')
        current=self.env
        while current is not raw:
            if 'Video' in type(current).__name__:raise ValueError('Video recording needs intermediate frames')
            current=current.env
        names=[n+'_image' for n in raw.camera_names]
        observables=tuple(raw.env._observables[n] for n in names)
        guard=raw.__dict__.get('_last_frame_rgb_guard')
        if guard is None or any(a is not b for a,b in zip(guard.observables,observables)):
            guard=Guard(raw);raw._last_frame_rgb_guard=guard;guards.append(guard)
        steps=guard.chunk_steps(self);raw._last_frame_rgb_schedule=[0,steps];guard.chunks+=1
        try:
            result=old_chunk(self,actions)
            if raw._last_frame_rgb_schedule[0]!=steps:raise ValueError('Unexpected early termination; cannot skip intermediate RGB')
            return result
        finally:
            raw._last_frame_rgb_schedule=None;guard.skip=False
    def control(self,action):
        schedule=self.__dict__.get('_last_frame_rgb_schedule')
        if schedule is None:return old_control(self,action)
        guard=self._last_frame_rgb_guard;schedule[0]+=1;guard.skip=schedule[0]<schedule[1]
        before=[c.rendered for c in guard.cameras]
        result=old_control(self,action);guard.steps+=1
        if not guard.skip and any(c.rendered<=n for c,n in zip(guard.cameras,before)):
            raise ValueError('Final RGB was not sampled at its native time')
        return result
    MultiStepWrapper.step=chunk;RoboCasaGymEnv.step=control
    def counts(selected):
        return {'control_steps':sum(g.steps for g in selected),'chunks':sum(g.chunks for g in selected),
                'rgb_rendered':sum(c.rendered for g in selected for c in g.cameras),
                'rgb_reused':sum(c.reused for g in selected for c in g.cameras)}
    def report():
        current=counts(guards)
        return {'version':'native_last_frame_rgb_v1',**{k:retired[k]+current[k] for k in retired}}
    def release(wrapper):
        raw=wrapper.unwrapped
        guard=raw.__dict__.get('_last_frame_rgb_guard')
        if guard is None:return
        if raw.__dict__.get('_last_frame_rgb_schedule') is not None:raise RuntimeError('Cannot release an active RGB chunk')
        if guard not in guards:raise RuntimeError('RGB guard belongs to another installation')
        values=counts([guard])
        for key in retired:retired[key]+=values[key]
        for observable,camera in zip(guard.observables,guard.cameras):
            if observable._sensor is not camera:raise RuntimeError('Camera sensor changed while RGB guard was active')
            observable._sensor=camera.original
        guards.remove(guard)
        del raw._last_frame_rgb_guard
    def close():
        if guards:raise RuntimeError('Release all RGB environments before uninstalling')
        if MultiStepWrapper.step is not chunk or RoboCasaGymEnv.step is not control:raise RuntimeError('RGB methods changed unexpectedly')
        MultiStepWrapper.step=old_chunk;RoboCasaGymEnv.step=old_control
    report.release=release
    report.close=close
    report.active_count=lambda:len(guards)
    return report
