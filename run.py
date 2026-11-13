"""Single-run and parallel parameter-scan command line interface."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import numba
import numpy as np
from model import advance, seed_events


@dataclass
class Config:
    length: int = 100
    r: float = 3.5
    threshold: int = 3
    beta: float = 0.1
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.01
    initial_cooperation: float = 0.5
    pressure: str = 'dependent'
    state: str = 'joint'
    mcs: int = 1000000
    tail: int = 100000
    progress_every: int = 10000
    record_every: int = 1
    seed: int = 2026
    snapshots: str = '0,3000,50000,1000000'


def simulate(config, folder):
    c = Config(**config)
    numeric = [c.r, c.beta, c.alpha, c.gamma, c.epsilon, c.initial_cooperation]
    if not all(np.isfinite(x) for x in numeric):
        raise ValueError('Parameters must be finite')
    if (c.length < 3 or not 0 <= c.threshold <= 5 or c.r < 0 or c.beta < 0
            or not 0 < c.alpha <= 1 or not 0 <= c.gamma < 1
            or not 0 <= c.epsilon <= 1 or not 0 <= c.initial_cooperation <= 1
            or not 1 <= c.tail <= c.mcs or min(c.progress_every, c.record_every) < 1
            or not 0 <= c.seed < 2**32):
        raise ValueError('Invalid parameter range; see README')
    labels = {'joint': ['CG', 'CB', 'DG', 'DB'], 'behavior': ['C', 'D'],
              'environment': ['G', 'B']}[c.state]
    representation = {'joint': 0, 'behavior': 1, 'environment': 2}[c.state]
    if c.pressure not in ('dependent', 'independent'):
        raise ValueError('Unknown pressure rule')
    snapshot_times = sorted(set(int(x) for x in c.snapshots.split(',') if x.strip()))
    if any(x < 0 for x in snapshot_times):
        raise ValueError('Snapshot times must be nonnegative')
    snapshot_times = [x for x in snapshot_times if x <= c.mcs]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)  # Refuse to overwrite a run.
    rng = np.random.default_rng(c.seed)
    actions = (rng.random((c.length, c.length)) < c.initial_cooperation).astype(np.int8)
    q = np.zeros((c.length, c.length, len(labels), 2))
    seed_events(c.seed)
    meta = dict(parameters=asdict(c), status='running', state_order=labels,
                action_order=['D', 'C'], python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('model.py', 'run.py')})
    def metadata():
        (folder / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    metadata()
    if 0 in snapshot_times:
        np.save(folder / 'actions_mcs_0.npy', actions)
    total = 0.0
    total2 = 0.0
    samples = 0
    started = time.perf_counter()
    with (folder / 'trajectory.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['mcs', 'rho_C'] + ['visits_' + s for s in labels] + ['A_' + s for s in labels])
        writer.writerow([0, float(actions.mean())] + [''] * (2 * len(labels)))
        completed = 0
        while completed < c.mcs:
            end = min(((completed // c.progress_every) + 1) * c.progress_every, c.mcs)
            future_snapshots = [x for x in snapshot_times if completed < x <= end]
            if future_snapshots:
                end = future_snapshots[0]
            rho, visits, sums = advance(actions, q, c.r, c.threshold, c.beta,
                c.alpha, c.gamma, c.epsilon, c.pressure == 'independent', representation, end - completed)
            for k, value in enumerate(rho):
                t = completed + k + 1
                if t > c.mcs - c.tail:
                    total += value
                    total2 += value * value
                    samples += 1
                if t % c.record_every == 0 or t == c.mcs:
                    advantage = np.divide(sums[k], visits[k], out=np.full(len(labels), np.nan), where=visits[k] > 0)
                    writer.writerow([t, value] + visits[k].tolist() + advantage.tolist())
            completed = end
            if end in snapshot_times:
                np.save(folder / f'actions_mcs_{end}.npy', actions)
            if end % c.progress_every == 0 or end == c.mcs:
                f.flush()
                meta.update(completed_mcs=end, elapsed_seconds=time.perf_counter() - started)
                metadata()
                print(f'{folder.name}: {end}/{c.mcs} MCS; rho_C={rho[-1]:.6f}', flush=True)
    np.savez_compressed(folder / 'final_state.npz', actions=actions, q=q)
    summary = dict(mean_cooperation=total / samples,
                   temporal_std_cooperation=float(np.sqrt(max(0.0, total2 / samples - (total / samples)**2))),
                   tail_mcs=samples, completed_mcs=c.mcs,
                   stationarity_assessed=False)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    meta.update(status='complete')
    metadata()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='examples/quickstart.json', help='JSON object or list of objects')
    parser.add_argument('--output', default='outputs/quickstart', help='New directory; existing runs are protected')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    data = json.loads(Path(args.config).read_text(encoding='utf8'))
    configs = data if isinstance(data, list) else [data]
    if not configs or args.workers < 1:
        parser.error('Provide at least one configuration and a positive worker count')
    # Validate keys before launching any job.
    for config in configs:
        Config(**config)
    root = Path(args.output)
    if root.exists():
        parser.error('Output directory already exists; choose a new name')
    jobs = [(config, root / f'run_{i:04d}') for i, config in enumerate(configs)]
    if args.workers == 1:
        for config, folder in jobs:
            simulate(config, folder)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(simulate, config, folder) for config, folder in jobs]
            for future in futures:
                future.result()


if __name__ == '__main__':
    main()
