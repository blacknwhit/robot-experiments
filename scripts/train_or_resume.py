"""Scheduler entry: fresh once, then full-state resume; no retry loop."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from common import ROOT


def command(config):
    config = Path(config).resolve(strict=True)
    cfg = yaml.safe_load(config.read_text())
    run = Path(cfg['run_root_dir']) / cfg['run_id']
    args = [sys.executable, str(ROOT / 'scripts/train.py'), '--config', str(config)]
    # A partial or failed first launch must never silently start from base again.
    # train_entry selects a complete compatible state, or fails with evidence.
    if run.exists():
        args.append('--resume')
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cmd = command(args.config)
    print(json.dumps({'command': cmd, 'automatic_retry_loop': False}), flush=True)
    if args.dry_run:
        cmd.append('--dry-run')
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
