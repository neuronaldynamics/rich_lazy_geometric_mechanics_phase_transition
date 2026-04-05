# MNIST FTLE vs Margin

Codebase for MNIST FTLE experiments and FTLE-vs-margin analysis. The project now supports both single-job runs and resumable manifest-driven batch sweeps.

## Layout

- `configs.py` — train/eval/plot/path dataclasses
- `paths.py` — canonical run naming and filesystem layout
- `runtime.py` — manifest I/O, immutable specs, status tracking, per-stage logging
- `train.py` — single-run training plus resumable job training
- `job_runner.py` — train/eval/plot stage execution for one job
- `build_manifest.py` — expand a sweep config into a JSONL manifest
- `run_batch.py` — main batch orchestrator
- `collect_results.py` — aggregate finished jobs into summary tables
- `configs/batch/*.yaml` — sample sweep definitions

## Output tree

Single-job CLI outputs are stored under `runs/single/`.

Batch jobs are stored under:

```text
runs/
  manifests/
  jobs/
    mnist/
      <job_id>/
        spec.json
        status.json
        logs/
        checkpoints/
        artifacts/
  summaries/
    <experiment_name>/
```

Each batch job folder contains:

- `spec.json` — immutable job configuration
- `status.json` — stage state machine (`pending`, `running`, `done`, `failed`)
- `logs/train.log`, `logs/eval.log`, `logs/plots.log`
- `checkpoints/latest.pt`, `checkpoints/best.pt`
- cached eval intermediates such as `artifacts/predictions.npz`, `artifacts/ftle_chunks/`, `artifacts/margin_chunks/`
- final outputs such as `artifacts/train_metrics.json`, `artifacts/eval_metrics.json`, `artifacts/ftle_margin_data.npz`, `artifacts/plots/*.png`

## Single-job workflow

Train:

```bash
python run_train.py --width 20 --depth 4 --gain 1.0 --lr 0.05 --seed 0
```

Evaluate:

```bash
python run_eval.py --width 20 --depth 4 --gain 1.0 --lr 0.05 --seed 0
```

Plot:

```bash
python run_plots.py --width 20 --depth 4 --gain 1.0 --lr 0.05 --seed 0
```

## Batch workflow

Build a manifest from a sweep config:

```bash
python build_manifest.py --config configs/batch/depth_sweep.yaml
```

Run the whole batch:

```bash
python run_batch.py --config configs/batch/depth_sweep.yaml
```

Collect summaries again later if needed:

```bash
python collect_results.py --config configs/batch/depth_sweep.yaml
```

`run_batch.py` will:

- create a manifest under `runs/manifests/`
- create per-job folders under `runs/jobs/mnist/`
- resume training from `checkpoints/latest.pt` if needed
- reuse completed FTLE and margin chunk files during evaluation
- skip stages already marked `done` when required outputs still exist
- write summary tables under `runs/summaries/<experiment_name>/`

## Batch config

The batch config is YAML with four sections:

- `sweep` — width/depth/gain/lr/seed grids
- `train` — batch size, max epochs, target accuracy
- `eval` — FTLE and adversarial-margin settings
- `plots` / `runtime` — plotting and failure policy

Sample configs included:

- `configs/batch/seed_stability.yaml`
- `configs/batch/depth_sweep.yaml`
- `configs/batch/width_sweep.yaml`

## Notes

- FTLE computation is exact in the current evaluation path and can be slow.
- Margin computation is cached in chunks so interrupted eval runs can resume.
- The batch runner is sequential today; the manifest and job layout are set up so parallel workers can be added later without changing the job format.
