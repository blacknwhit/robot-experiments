from pathlib import Path
import ast,hashlib,json,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'release_manifest.json')=='65c003c6766692a299eed7f58d075bbc7060e1ed8c0338f8bfec4c198a3c0dbb'
counts={}
for directory,manifest in [(ROOT,'release_manifest.json'),(HERE,'manifest.json')]:
    files=json.loads((directory/manifest).read_text())['files']
    for name,digest in files.items():
        p=directory/name;p.resolve().relative_to(directory)
        assert sha(p)==digest,name
        if p.suffix=='.py' and directory==HERE:ast.parse(p.read_text())
    counts[manifest]=len(files)
proof=dict(verified_at=time.time(),counts=counts,training_release_unchanged=True,eval_revision_sha256=sha(HERE/'manifest.json'))
(HERE/'deploy_verified.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
