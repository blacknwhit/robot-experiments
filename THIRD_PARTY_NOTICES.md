# Third-party notices

This repository preserves source attribution and license notices. It does not relicense third-party code or redistribute training datasets/model weights.

| Component | Source | Terms retained here |
|---|---|---|
| StarVLA and experiment adaptations | https://github.com/starVLA/starVLA | Root LICENSE and each runtime/LICENSE; existing source headers |
| OpenPI, openpi-client | https://github.com/Physical-Intelligence/openpi and https://github.com/robocasa-benchmark/openpi | Each `runtimes/*/playground/Code/openpi_robocasa365/LICENSE` (Apache-2.0) and `LICENSE_GEMMA.txt` |
| Transformers patch files | OpenPI's pinned `models_pytorch/transformers_replace`; upstream Hugging Face Transformers | Original copyright/Apache headers preserved; full installed Transformers is generated locally, not vendored |
| RoboCasa training registry | https://github.com/robocasa/robocasa commit `6f90063c0ec2b9916bbd977368b54c98d9c2d908` | `assets/ROBOCASA_LICENSE` |
| RoboCasa-compatible simulator wrappers | StarVLA/NVIDIA source | Apache-2.0 notices in each wrapper |

The downloaded RoboCasa/NVIDIA datasets, kitchen/robot assets, PI0.5/PaliGemma weights and tokenizer have their own upstream license/terms. A code repository license is not permission to ignore those terms. Download scripts fetch from the original public hosts; review their model/data cards before redistribution or commercial use.

Small normalization statistics and task definitions are included solely as preprocessing/protocol artifacts with their source identifiers in the handoff. User-trained parameters and operational credentials are not included.
