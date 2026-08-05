# Modified-files-only patch

Base: `Tiara2-v2.3.0-bootstrap-baseline-freeze.zip`

- `tiara/__init__.py`: lazy legacy imports so schema/config tools do not require the ML stack.
- `tiara/hierarchical/`: schema, strict labels, feature preparation, shared encoder/multi-head model, masked loss, trainer, cascade inference and evaluation.
- `config/config_v2_3_0_hierarchical.yaml`: v2.3.0 config plus reserved v2.3.1/v2.3.2 hooks.
- `tests/hierarchical/`: schema/label/model tests.
- `apply_patch.sh`: copies only these changed files into the bootstrap repository.
