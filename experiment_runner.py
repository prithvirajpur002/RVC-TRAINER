#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Catalyst RVC — Experiment Runner
STRICT experiment control. Every run is fully traceable.

Requirements:
  - exp_id: unique identifier (exp_001, exp_002, etc.)
  - dataset: MUST be one of: clean, natural, raw
  - config:  MUST be one of: baseline, high_quality
  - epochs, batch_size: exact values
  - changed_from: which experiment this builds on (required for exp_002+)
  - change_note:  brief human note on why (e.g. "testing natural dataset")

Storage structure (IMMUTABLE after creation):
  experiments/
    exp_001/
      config.json           ← defines EXACTLY what ran + what changed
      model/
      samples/
      logs/
    decision_log.json       ← records of every winner decision

Fixes (vs previous version):
  1. ExperimentSpec now includes changed_from + change_note fields
  2. ExperimentValidator stamps changelog into config.json automatically
  3. 'decide' command writes to decision_log.json (enforced before next exp)
  4. 'suggest' command recommends what to change next based on winner
  5. Decision is REQUIRED before creating exp_002+ (enforced)
  6. ExperimentSpec now records rvc_commit (git hash of RVC repo) so results
     from exp_001 and exp_003 are still comparable even if RVC updated in between.
     Auto-detected at run time; falls back to "unknown" if git is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from src.utils                import Logger, elapsed_str, load_json, now_iso, save_json
from src.experiment_validator import ExperimentValidator


def _get_rvc_commit(rvc_repo_path: str) -> str:
    """
    Auto-detect the current git commit hash of the RVC repo.

    Why this matters: if the RVC repo updates between exp_001 and exp_003,
    you're no longer comparing apples to apples. Storing the commit hash in
    config.json makes this immediately visible.

    Returns "unknown" if git is unavailable or the path is not a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=rvc_repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # short hash, readable
    except Exception:
        pass
    return "unknown"


class ExperimentSpec:
    """Defines an experiment. Immutable once written to config.json."""

    VALID_DATASETS = {"clean", "natural", "raw"}
    VALID_CONFIGS  = {"baseline", "high_quality"}

    def __init__(
        self,
        exp_id:           str,
        dataset:          str,
        config:           str,
        epochs:           int,
        batch_size:       int,
        changed_from:     Optional[str] = None,
        change_note:      str = "",
        rvc_commit:       str = "",
        changed_variable: str = "",   # which single field changed vs changed_from
    ):
        if not exp_id or not exp_id.startswith("exp_"):
            raise ValueError(f"exp_id must start with 'exp_', got: {exp_id!r}")
        if dataset not in self.VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {self.VALID_DATASETS}, got: {dataset!r}"
            )
        if config not in self.VALID_CONFIGS:
            raise ValueError(
                f"config must be one of {self.VALID_CONFIGS}, got: {config!r}"
            )
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got: {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got: {batch_size}")

        self.exp_id           = exp_id
        self.dataset          = dataset
        self.config           = config
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.changed_from     = changed_from   # which experiment this builds on
        self.change_note      = change_note    # brief human note
        self.rvc_commit       = rvc_commit     # git commit of RVC repo at run time
        self.changed_variable = changed_variable  # which single field changed

    def to_dict(self) -> dict:
        d = {
            "exp_id":     self.exp_id,
            "dataset":    self.dataset,
            "config":     self.config,
            "epochs":     self.epochs,
            "batch_size": self.batch_size,
            "rvc_commit": self.rvc_commit or "unknown",
        }
        if self.changed_from:
            d["changed_from"] = self.changed_from
        if self.changed_variable:
            d["changed_variable"] = self.changed_variable
        if self.change_note:
            d["change_note"] = self.change_note
        return d

    @staticmethod
    def from_dict(d: dict) -> "ExperimentSpec":
        return ExperimentSpec(
            exp_id            = d["exp_id"],
            dataset           = d["dataset"],
            config            = d["config"],
            epochs            = d["epochs"],
            batch_size        = d["batch_size"],
            changed_from      = d.get("changed_from"),
            change_note       = d.get("change_note", ""),
            rvc_commit        = d.get("rvc_commit", ""),
            changed_variable  = d.get("changed_variable", ""),
        )


class ExperimentRunner:
    """Strict control over experiment execution and iteration."""

    def __init__(self, base_dir: str):
        self.base_dir  = base_dir
        self.exp_dir   = os.path.join(base_dir, "experiments")
        self.log       = Logger()
        self.validator = ExperimentValidator(self.exp_dir)
        os.makedirs(self.exp_dir, exist_ok=True)

    # ── Creation ──────────────────────────────────────────────────────────────

    def create_experiment(self, spec: ExperimentSpec) -> str:
        """
        Create experiment directory structure. Fails if exp_id already exists.

        Enforces:
          1. A decision must be recorded for the previous experiment before
             creating this one (except for exp_001).
          2. If changed_from is given, exactly ONE variable must differ.
        """
        exp_path = os.path.join(self.exp_dir, spec.exp_id)

        if os.path.exists(exp_path):
            raise RuntimeError(
                f"Experiment {spec.exp_id} already exists at {exp_path}.\n"
                "Cannot redefine. Create a new experiment ID instead."
            )

        # ── Enforce: decision required before creating next exp ───────────────
        can_proceed, reason = self.validator.check_decision_required(
            self.exp_dir, spec.exp_id
        )
        if not can_proceed:
            raise RuntimeError(f"Decision gate: {reason}")

        # ── Enforce: single-variable change when changed_from is given ────────
        if spec.changed_from:
            prev_spec = self.validator.get_spec(spec.changed_from)
            if prev_spec:
                current_dict = spec.to_dict()
                changes = [
                    k for k in ["dataset", "config", "epochs", "batch_size"]
                    if prev_spec.get(k) != current_dict.get(k)
                ]
                if len(changes) > 1:
                    raise RuntimeError(
                        f"Too many changes from {spec.changed_from}: "
                        f"{', '.join(changes)}. Change exactly ONE variable."
                    )
                if len(changes) == 0:
                    raise RuntimeError(
                        f"This experiment is identical to {spec.changed_from}. "
                        "Change exactly ONE variable."
                    )
                # Store which variable changed so print_summary and the UI both see it
                spec.changed_variable = changes[0]

        # ── Create directories ────────────────────────────────────────────────
        os.makedirs(exp_path, exist_ok=True)
        for subdir in ["model", "samples", "logs"]:
            os.makedirs(os.path.join(exp_path, subdir), exist_ok=True)

        # Write config.json
        save_json(os.path.join(exp_path, "config.json"), spec.to_dict())

        # Stamp changelog into config.json automatically
        if spec.changed_from:
            self.validator.stamp_changelog(spec.changed_from, spec.exp_id)

        self.log.ok(f"Created: {exp_path}")
        if spec.changed_from:
            self.log.info(
                f"  ↳ Building on {spec.changed_from}"
                + (f": {spec.change_note}" if spec.change_note else "")
            )

        return exp_path

    # ── Execution ─────────────────────────────────────────────────────────────

    def run_experiment(self, spec: ExperimentSpec) -> dict:
        """Run experiment. Must have been created first."""
        exp_path = os.path.join(self.exp_dir, spec.exp_id)

        if not os.path.exists(exp_path):
            return {
                "success": False,
                "exp_id":  spec.exp_id,
                "error":   "Experiment not created. Call create_experiment() first.",
            }

        logs_dir = os.path.join(exp_path, "logs")
        log_file = os.path.join(logs_dir, "run.log")
        status_path = os.path.join(exp_path, "status.json")
        os.makedirs(logs_dir, exist_ok=True)

        main_script = os.path.join(_here, "main.py")
        if not os.path.exists(main_script):
            return {
                "success": False,
                "exp_id":  spec.exp_id,
                "error":   f"main.py not found at {main_script}",
            }

        # Auto-detect RVC commit and stamp it into the spec before running
        rvc_repo_path = os.path.join(self.base_dir, "Retrieval-based-Voice-Conversion-WebUI")
        if not os.path.isdir(rvc_repo_path):
            # Try common alternate locations
            for candidate in [
                os.path.join(self.base_dir, "rvc"),
                os.path.join(self.base_dir, "rvc-repo"),
                "/kaggle/working/Retrieval-based-Voice-Conversion-WebUI",
            ]:
                if os.path.isdir(candidate):
                    rvc_repo_path = candidate
                    break

        rvc_commit = _get_rvc_commit(rvc_repo_path)
        if not spec.rvc_commit or spec.rvc_commit == "unknown":
            spec.rvc_commit = rvc_commit
            # Update config.json with the detected commit
            config_path = os.path.join(exp_path, "config.json")
            if os.path.exists(config_path):
                try:
                    cfg = load_json(config_path)
                    cfg["rvc_commit"] = rvc_commit
                    save_json(config_path, cfg)
                except Exception:
                    pass

        self.log.section(f"RUNNING: {spec.exp_id}")
        self.log.info(
            f"  dataset={spec.dataset}  config={spec.config}  "
            f"epochs={spec.epochs}  batch={spec.batch_size}"
        )
        self.log.info(f"  rvc_commit={spec.rvc_commit}")
        if spec.changed_from:
            self.log.info(f"  changed_from={spec.changed_from}: {spec.change_note}")

        # Capture the real start time NOW, before the subprocess runs.
        # Previously this was written as "started_at": status_path (a file
        # path string), silently corrupting every completed experiment record.
        started_at = now_iso()

        save_json(status_path, {
            "status":     "running",
            "started_at": started_at,
            "spec":       spec.to_dict(),
        })

        start = time.time()
        try:
            result = subprocess.run(
                [
                    sys.executable, main_script,
                    "--only", spec.exp_id,
                ],
                cwd=_here,
                capture_output=False,
            )

            elapsed  = elapsed_str(start)
            success  = result.returncode == 0
            status   = "complete" if success else "failed"

            save_json(status_path, {
                "status":       status,
                "started_at":   started_at,
                "completed_at": now_iso(),
                "elapsed":      elapsed,
                "return_code":  result.returncode,
                "spec":         spec.to_dict(),
            })

            if success:
                self.log.ok(f"{spec.exp_id} complete in {elapsed}")
            else:
                self.log.error(f"{spec.exp_id} FAILED (code {result.returncode})")

            return {
                "success":  success,
                "exp_id":   spec.exp_id,
                "elapsed":  elapsed,
                "log_file": log_file,
            }

        except Exception as e:
            save_json(status_path, {
                "status": "failed",
                "error":  str(e),
                "spec":   spec.to_dict(),
            })
            return {"success": False, "exp_id": spec.exp_id, "error": str(e)}

    # ── Decision recording ────────────────────────────────────────────────────

    def record_decision(
        self,
        winner_exp_id: str,
        loser_exp_id: Optional[str],
        reason: str,
        next_exp_rationale: str = "",
    ) -> None:
        """
        Record which experiment won and why.

        This is REQUIRED before you can create the next experiment.
        Writes to experiments/decision_log.json.
        """
        if not reason.strip():
            raise ValueError("reason cannot be empty. What did you learn?")

        log_path = os.path.join(self.exp_dir, "decision_log.json")
        if os.path.exists(log_path):
            try:
                log = load_json(log_path)
            except Exception:
                log = {"decisions": []}
        else:
            log = {"decisions": []}

        entry = {
            "winner_exp_id":       winner_exp_id,
            "loser_exp_id":        loser_exp_id,
            "reason":              reason,
            "next_exp_rationale":  next_exp_rationale,
            "recorded_at":         now_iso(),
        }
        log["decisions"].append(entry)
        log["latest_winner"] = winner_exp_id
        log["updated_at"]    = now_iso()

        save_json(log_path, log)
        self.log.ok(
            f"Decision recorded: {winner_exp_id} beat {loser_exp_id or 'baseline'}"
        )
        self.log.info(f"  Reason: {reason}")

        # Print suggestion for next experiment
        suggestion = self.validator.get_recommendation(winner_exp_id)
        self.log.info(f"\nNext step:\n  {suggestion}")

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_experiment_info(self, exp_id: str) -> Optional[dict]:
        """Get full info about an experiment."""
        exp_path = os.path.join(self.exp_dir, exp_id)
        if not os.path.exists(exp_path):
            return None

        spec_dict   = self.validator.get_spec(exp_id)
        status_path = os.path.join(exp_path, "status.json")
        status      = {}
        if os.path.exists(status_path):
            try:
                status = load_json(status_path)
            except Exception:
                pass

        return {
            "exp_id":   exp_id,
            "path":     exp_path,
            "spec":     spec_dict,
            "status":   status.get("status", "unknown"),
            "elapsed":  status.get("elapsed"),
        }

    def list_experiments(self) -> list[dict]:
        """List all experiments sorted by ID."""
        if not os.path.exists(self.exp_dir):
            return []
        exps = []
        for d in sorted(os.listdir(self.exp_dir)):
            if not d.startswith("exp_"):
                continue
            info = self.get_experiment_info(d)
            if info:
                exps.append(info)
        return exps

    def print_summary(self) -> None:
        """Print a human-readable summary of all experiments."""
        exps = self.list_experiments()
        if not exps:
            self.log.warn("No experiments found.")
            return

        self.log.section("EXPERIMENT SUMMARY")
        header = (
            f"{'ID':<12} {'Dataset':<10} {'Config':<14} "
            f"{'Epochs':>6} {'Batch':>5}  {'Status':<10}  {'Changed from'}"
        )
        self.log.info(header)
        self.log.info("─" * len(header))

        for exp in exps:
            spec   = exp.get("spec") or {}
            status = exp.get("status", "unknown")
            cf     = spec.get("changed_from", "—")
            cv     = spec.get("changed_variable", "")
            note   = spec.get("change_note", "")
            change_str = f"{cf} [{cv}]" if cv else cf

            self.log.info(
                f"  {exp['exp_id']:<10}  {spec.get('dataset','?'):<8}  "
                f"{spec.get('config','?'):<12}  "
                f"{spec.get('epochs','?'):>5}  "
                f"{spec.get('batch_size','?'):>4}  "
                f"{status:<10}  {change_str}"
                + (f"  — {note}" if note else "")
            )

        # Print decision log summary
        log_path = os.path.join(self.exp_dir, "decision_log.json")
        if os.path.exists(log_path):
            try:
                dlog     = load_json(log_path)
                decisions = dlog.get("decisions", [])
                self.log.info(f"\nDecisions recorded: {len(decisions)}")
                for d in decisions[-3:]:  # show last 3
                    self.log.info(
                        f"  {d['winner_exp_id']} beat {d.get('loser_exp_id','?')}"
                        f" — {d['reason'][:80]}"
                    )
            except Exception:
                pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Catalyst RVC — experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  create   exp_001 clean baseline 200 6         Create a new experiment
  run      exp_001                               Run a created experiment
  decide   exp_001 --winner exp_001 --reason '…' Record decision after comparing
  suggest  exp_001                               Get suggestion for next change
  info     exp_001                               Show experiment details
  list                                           List all experiments
  summary                                        Print full summary table

examples:
  # First experiment — no changed_from needed
  python experiment_runner.py create exp_001 clean baseline 200 6

  # Second experiment — must change exactly one variable, must state what
  python experiment_runner.py create exp_002 natural baseline 200 6 \\
    --changed-from exp_001 --note "testing natural dataset vs clean"

  # Record decision before creating exp_003
  python experiment_runner.py decide \\
    --winner exp_002 --loser exp_001 \\
    --reason "natural dataset sounded more human on test_1"

  # Get suggestion for what to change next
  python experiment_runner.py suggest exp_002
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── create ────────────────────────────────────────────────────────────────
    cp = sub.add_parser("create", help="Create a new experiment")
    cp.add_argument("exp_id")
    cp.add_argument("dataset",    choices=list(ExperimentSpec.VALID_DATASETS))
    cp.add_argument("config",     choices=list(ExperimentSpec.VALID_CONFIGS))
    cp.add_argument("epochs",     type=int)
    cp.add_argument("batch_size", type=int)
    cp.add_argument("--changed-from", dest="changed_from", default=None)
    cp.add_argument("--note",        dest="change_note",  default="")
    cp.add_argument("--rvc-commit",  dest="rvc_commit",   default="",
                    help="Git commit of RVC repo (auto-detected at run time if omitted)")

    # ── run ───────────────────────────────────────────────────────────────────
    rp = sub.add_parser("run", help="Run a created experiment")
    rp.add_argument("exp_id")

    # ── decide ────────────────────────────────────────────────────────────────
    dp = sub.add_parser("decide", help="Record a decision after comparing experiments")
    dp.add_argument("--winner",  required=True)
    dp.add_argument("--loser",   default=None)
    dp.add_argument("--reason",  required=True)
    dp.add_argument("--next-rationale", dest="next_rationale", default="")

    # ── suggest ───────────────────────────────────────────────────────────────
    sgp = sub.add_parser("suggest", help="Get suggestion for what to change next")
    sgp.add_argument("exp_id", help="Winner experiment to base suggestion on")

    # ── info ──────────────────────────────────────────────────────────────────
    ip = sub.add_parser("info", help="Show experiment details")
    ip.add_argument("exp_id")

    # ── list / summary ────────────────────────────────────────────────────────
    sub.add_parser("list",    help="List all experiments")
    sub.add_parser("summary", help="Print full summary table with decisions")

    args    = parser.parse_args()
    runner  = ExperimentRunner("/kaggle/working/catalyst_rvc")

    if args.command == "create":
        try:
            spec = ExperimentSpec(
                exp_id       = args.exp_id,
                dataset      = args.dataset,
                config       = args.config,
                epochs       = args.epochs,
                batch_size   = args.batch_size,
                changed_from = args.changed_from,
                change_note  = args.change_note,
                rvc_commit   = args.rvc_commit,   # empty → stamped at run time
                # changed_variable is computed inside create_experiment()
            )
            runner.create_experiment(spec)
            print(json.dumps({"success": True, "exp_id": args.exp_id, "changed_variable": spec.changed_variable}))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
            sys.exit(1)

    elif args.command == "run":
        info = runner.get_experiment_info(args.exp_id)
        if not info or not info.get("spec"):
            print(json.dumps({"success": False, "error": f"{args.exp_id} not found"}))
            sys.exit(1)
        spec   = ExperimentSpec.from_dict(info["spec"])
        result = runner.run_experiment(spec)
        print(json.dumps(result))

    elif args.command == "decide":
        try:
            runner.record_decision(
                winner_exp_id       = args.winner,
                loser_exp_id        = args.loser,
                reason              = args.reason,
                next_exp_rationale  = args.next_rationale,
            )
            print(json.dumps({"success": True}))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
            sys.exit(1)

    elif args.command == "suggest":
        suggestion = runner.validator.get_recommendation(args.exp_id)
        print(suggestion)

    elif args.command == "info":
        info = runner.get_experiment_info(args.exp_id)
        if not info:
            print(json.dumps({"error": f"{args.exp_id} not found"}))
            sys.exit(1)
        print(json.dumps(info, indent=2, default=str))

    elif args.command == "list":
        exps = runner.list_experiments()
        for exp in exps:
            spec = exp.get("spec") or {}
            cf   = spec.get("changed_from", "—")
            cv   = spec.get("changed_variable", "")
            print(
                f"{exp['exp_id']:12} {spec.get('dataset','?'):8} "
                f"{spec.get('config','?'):12} "
                f"epochs={spec.get('epochs','?'):3} "
                f"batch={spec.get('batch_size','?')} "
                f"[{exp.get('status','?')}]"
                + (f"  ← {cf} [{cv}]" if cv else "")
            )

    elif args.command == "summary":
        runner.print_summary()


if __name__ == "__main__":
    main()
