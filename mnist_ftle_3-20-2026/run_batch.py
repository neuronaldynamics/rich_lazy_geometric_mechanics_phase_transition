from __future__ import annotations

import argparse
from pathlib import Path

from build_manifest import build_manifest
from collect_results import collect_results
from configs import PathConfig
from job_runner import run_job
from paths import ensure_dirs
from runtime import load_jsonl, resolve_job_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()

    if args.config is None and args.manifest is None:
        raise ValueError("Provide --config or --manifest.")

    paths = PathConfig()
    ensure_dirs(paths)

    manifest_path = args.manifest
    if args.config is not None:
        manifest_path = build_manifest(args.config, paths)
    assert manifest_path is not None

    jobs = load_jsonl(manifest_path)
    print(f"[batch] manifest={manifest_path} jobs={len(jobs)}")
    for spec in jobs:
        job_path = resolve_job_path(paths, spec)
        continue_on_error = bool(spec.get("runtime", {}).get("continue_on_error", True))
        print(f"[batch] running job {spec['job_id']}")
        run_job(job_path=job_path, spec=spec, continue_on_error=continue_on_error)

    summary_dir = None
    if jobs:
        summary_dir = collect_results(
            experiment_name=jobs[0]["experiment_name"],
            paths=paths,
            dataset=jobs[0].get("dataset", "mnist"),
            job_ids=[spec["job_id"] for spec in jobs],
        )
    print({"manifest": str(manifest_path), "jobs": len(jobs), "summary_dir": str(summary_dir) if summary_dir else None})


if __name__ == "__main__":
    main()
