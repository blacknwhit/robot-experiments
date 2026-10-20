"""Strict action-only server, native full state load, no global BF16 cast."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NVIDIA_TF32_OVERRIDE','0')
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import pyarrow
import argparse,json,logging
from pathlib import Path
from common import activate,VARIANTS,POSITION_VARIANTS,is_point_variant,sha256
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args();activate(a.variant)
    from examples.pi05_robocasa365.eval_files import strict_pytorch_policy_server as strict
    from starVLA.model.framework.base_framework import build_framework
    from omegaconf import OmegaConf
    from strict_state_loader import strict_full_state_load
    from safetensors import safe_open
    cfg=OmegaConf.load(a.config);cfg.framework.pi05.init_checkpoint=None
    class CheckedWrapper(strict.PolicyServerWrapper):
        def __init__(self,ckpt_path,device='cuda',use_bf16=False,**kw):
            if use_bf16:raise ValueError('Blanket BF16 cast is not this evaluation protocol')
            self._ckpt_path=str(ckpt_path);framework=build_framework(cfg)
            with safe_open(str(ckpt_path),framework='pt',device='cpu') as f:
                wrapped=all(k.startswith('policy_model.') for k in f.keys())
            proof=strict_full_state_load(framework if wrapped else framework.policy_model,ckpt_path)
            self._framework=framework.to(device).eval();self._model_cfg=OmegaConf.to_container(cfg,resolve=True)
            self._available_unnorm_keys=[];self._default_unnorm_key=None;self._norm_processors={}
            self._direct_environment_actions=True;self._action_chunk_size=50
            if framework.policy_metadata['action_output_space']!='environment':raise ValueError('Wrong action space')
            if (framework.action_horizon,framework.environment_action_dim,framework.environment_state_dim,framework.num_inference_steps)!=(50,12,16,10):raise ValueError('Policy dimensions differ')
            model=framework.policy_model
            if is_point_variant(a.variant):
                expected = 'relative_position_and_difference_v3' if a.variant in POSITION_VARIANTS else 'adjacent_and_cumulative_v2'
                if model.point_track.objective_version!=expected:raise ValueError('Wrong trajectory objective')
                def forbidden(*args):raise RuntimeError('Point head executed at inference')
                model.point_track_head.register_forward_pre_hook(forbidden)
                if a.variant=='query3d_v2' and tuple(model.future_queries.shape)!=(1,50,2048):raise ValueError('Wrong Query50 shape')
                if (a.variant=='embedding3d_v2' or a.variant in POSITION_VARIANTS) and model.foresight.enabled:raise ValueError('Embedding must retain original prefix')
                if a.variant in POSITION_VARIANTS:
                    layers = 0 if a.variant == 'embedding3d_pos_a' else 2
                    if model.point_track.temporal_layers != layers or model.point_track.temporal_rope:
                        raise ValueError('Wrong V3 temporal decoder contract')
            elif a.variant=='query_fixed':
                if tuple(model.future_queries.shape)!=(1,64,2048):raise ValueError('Wrong FIXED queries')
            self._proof={'variant':a.variant,'checkpoint_sha256':sha256(ckpt_path),'config_sha256':sha256(a.config),'strict_full_state':proof}
        @property
        def metadata(self):return {**super().metadata,**self._proof}
    strict.PolicyServerWrapper=CheckedWrapper
    args=strict.build_argparser().parse_args(['--ckpt_path',a.checkpoint,'--config_yaml',a.config,'--port',str(a.port)])
    logging.basicConfig(level=logging.INFO,force=True);strict.main(args)
if __name__=='__main__':main()
