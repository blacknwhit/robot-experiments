"""An evaluation-only candidate; training allocator policy is untouched."""
ALLOCATOR_KEYS = ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')
CASES = ('a_original', 'a_default_allocator', 'baseline_reference')

def case_environment(parent, case):
    if case not in CASES:
        raise ValueError(case)
    env = dict(parent)
    if case != 'a_original':
        for key in ALLOCATOR_KEYS:
            env.pop(key, None)
    env['REFERENCE_EVAL_CASE'] = case
    return env
