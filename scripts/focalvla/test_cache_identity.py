"""Run on the target shared filesystem: concurrent cold initialization/resume."""
import concurrent.futures,hashlib,json,multiprocessing,tempfile
from pathlib import Path
from prepare_fast import cache_identity


def worker(args):
    root,resources=args
    return cache_identity(resources,Path(root))


def main():
    with tempfile.TemporaryDirectory(prefix='identity_test_',dir=Path(__file__).resolve().parent) as tmp:
        root=Path(tmp);weights=root/'weights';weights.mkdir()
        (root/'PLAN.json').write_text(json.dumps({'contract':'concurrency_fixture'}))
        for name in ['vggt','track4world','model.safetensors']:(weights/name).write_bytes((name*1000).encode())
        resources={'vggt_weights':str(weights/'vggt'),'track4world_weights':str(weights/'track4world'),'da3_weights':str(weights)}
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            cold=list(pool.map(worker,[(str(root),resources)]*4))
            before=(root/'CACHE_IDENTITY.json').stat().st_mtime_ns
            warm=list(pool.map(worker,[(str(root),resources)]*4))
        assert all(v==cold[0] for v in cold+warm)
        assert before==(root/'CACHE_IDENTITY.json').stat().st_mtime_ns
        assert cold[0]['weights']['vggt_weights']==hashlib.sha256((weights/'vggt').read_bytes()).hexdigest()
        print(json.dumps({'passed':True,'processes':4,'cold_and_resume':True,'identity_unchanged_on_resume':True}))


if __name__=='__main__':main()
