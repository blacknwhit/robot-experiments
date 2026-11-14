"""The existing PI0.5 LeRobot modality order, applied to raw parquet arrays."""
import numpy as np
STATE_KEYS=['end_effector_position_relative','end_effector_rotation_relative','base_position','base_rotation','gripper_qpos']
ACTION_KEYS=['end_effector_position','end_effector_rotation','gripper_close','base_motion','control_mode']
def repack(raw,modality,kind):
 keys=STATE_KEYS if kind=='state' else ACTION_KEYS
 return np.concatenate([np.asarray(raw)[...,modality[kind][k]['start']:modality[kind][k]['end']] for k in keys],axis=-1).astype(np.float32)
