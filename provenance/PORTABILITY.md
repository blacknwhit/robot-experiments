# Source and portability record

Packaged on 2026-10-19. Historical scores are not new GPU validations of this package.

| Runtime | Main source | OpenPI source |
|---|---|---|
| baseline | `415e087b89004425d5fd4292ab6a588d2e85bcb4` | `82c3c12f3c1524092bf948d3ced8e5cc7456a210` |
| query_fixed | reconstructed from `a11939afb38344603345479efa32aa5f3e8e4fc2`, plus audited trainer | `b18c13ae06e05231bb52387989b691b783c57070` |
| trajectory_v2 | complete source closure above, overlaid with deployed 2026-10-16 trajectory model/trainer/data sources | b18 tree plus deployed trajectory/Pri4R/augmentation/cache changes |

Baseline source archive matched all26 independently stored critical source files (16main +10OpenPI). FIXED's original training main ID was `a8e9d4b31ad3801243cba9545d80130aa19542f7`; the exact commit object was unavailable locally. The full fallback tree, archived runtime patch and accepted evaluation overlays matched82/82 historical evaluation manifest files before packaging. This package reuses those scientific implementations and replaces the old deployment orchestration. It is not a claim that every published file has the old SHA or that the historical stochastic training path has been re-executed.

Portability changes:

1. Separate runtime trees and explicit, resolved paths; old private filesystem fallbacks replaced with an unmistakable UNCONFIGURED sentinel. Old scheduler scripts, credentials, callback receipts, monitors and results databases are not shipped.
2. Complete scientific config templates; new global64 world-size resolution. Main training remains native VLATrainer/ForesightTrainer/QueryPointTrainer. Baseline receives a separate stop gate without changing the100k scheduler horizon.
3. Added full save after update1 and at a non-interval requested stop, checked same-world native state selection, rank0 selection broadcast, and a launcher writer lock. Native Adam/model/RNG/scheduler/progress format remains intact. No automatic checkpoint pruning is enabled.
4. FIXED/baseline use the later audited restore checker for actual world size and ZeRO padding. This checks loaded tensors; it does not reshard or alter them. Cross-world resume is not supported by this portable launcher.
5. Rebuild local Transformers4.53.2 overlays per runtime. Preserve upstream lock and record explicit CUDA128/NCCL/dependency overrides in setup and freeze receipts.
6. Reconstruct GT from immutable public source states, preserving original sampling and objectives; new root/code identity produces new GT provenance hashes. Reject incomplete/misaligned GT rather than silently filtering samples.
7. Common strict evaluation wrapper with full key/tied-alias/value checks, selective native precision, policy SHA/config/variant handshake, action-only inference guard for3D, and original task/horizon/seed/replan/trace protocol. One-task-per-GPU orchestration is scheduler-neutral; no SCO operational IDs are embedded.
8. New package tests exercise recipe contracts, disjoint4/2GPU coverage, incomplete-checkpoint skipping, wrong-world rejection and raw-result checks. GPU/closed-loop acceptance must still run on the destination cluster.

Frozen external identities:

- Training registry: RoboCasa `6f90063c0ec2b9916bbd977368b54c98d9c2d908`, SHA256 `23e8e31bb56bae80de86c34eafadfe32dd1e1eba6608376152ba9fdda5532a4b`.
- Simulator: RoboCasa `8f3c96ec8d1bfcd8126cad2bca887da98d30e997` (v1.0); robosuite `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`. The latter is a traceable new pin, not a recovered historical deployment SHA.
- Norm: RoboCasa checkpoint repository revision `916d8ec4aab4e6f930ba6b5b9bb8537ad16d4bb8`, pi05_pretrain_human300/75000 assets; SHA `4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee`.
- Clean converted base historical SHA: `9fd36e337832960bbfc122bd686dfc64b89551522d891c0d21ae586385de297e`.
- Best FIXED70k historical weights SHA: `fa5c5078afb8efa807a7af6129ff969ab1c0c4ccebf9e10180bb18c2a1d855a2`.

`source_files.sha256` identifies the exact published runtime/config/script/assets files. It excludes itself, Git metadata, downloaded resources and mutable validation reports.
