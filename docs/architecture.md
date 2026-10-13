# Module Layout

```text
PMV-CRN-GitHub/
  pmv_crn/
    models/
      common.py
      temporal.py
      image.py
      image_variants.py
      concept_rule.py
      fusion.py
      network.py
      variants.py
      api.py
    data/
      representations.py
      features.py
      splits.py
      dataset.py
    training/
      losses.py
      batch.py
      engine.py
      balanced.py
      entrypoint.py
    evaluation/
      metrics.py
      inference.py
    visualization/
      analysis.py
    baselines/
      models.py
      data.py
      evaluation.py
      training.py
      plots.py
      runner.py
    config/
      cli.py
    utils/
      random.py
  scripts/
  configs/
    refactor_manifest.json
  tests/
  docs/
  pyproject.toml
  requirements.txt
  README.md
```

Data construction lives in `data`; model layers and networks live in `models`; training objectives and orchestration live in `training`. Prediction and threshold metrics are separate from visualization. Baseline models have their own data, evaluation, training, and plotting modules.

The `configs` directory holds the refactor integrity manifest, not a newly invented experiment protocol. Runtime settings remain in the existing CLI parser. No training defaults have been changed.

The root-level legacy modules redirect to the corresponding package modules. The `tmpmodel` redirect preserves module identity so existing balanced-training overrides still affect the training engine's model and split function. Legacy command-line execution remains supported.

The model attributes, state-dictionary keys, function bodies, and training objectives have not been redesigned. Module paths change under the package layout; compatibility imports preserve the original public entry points. A checkpoint runtime loading test still requires the original training environment and authorized checkpoints.

See `README.md` for version status and implementation boundaries. This structural refactor does not resolve the previously identified experimental-protocol differences.
