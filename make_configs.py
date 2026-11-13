"""Generate paper parameter grids; generating a file does not run simulations."""
import argparse
import itertools
import json
from pathlib import Path


def grid(preset):
    rs = [x / 10 for x in range(28, 51)]
    betas = [0, 0.05, 0.1, 0.15, 0.2]
    fine_betas = [x / 100 for x in range(21)]
    if preset == 'fig2a':
        return [dict(r=r, beta=b) for r, b in itertools.product(rs, betas)]
    if preset == 'fig2b':
        return [dict(r=r, beta=b) for r, b in itertools.product([3.2, 3.4, 3.6, 3.8, 4.0], fine_betas)]
    if preset == 'fig3-4':
        return [dict(beta=b) for b in [0, 0.1, 0.2]]
    if preset == 'fig5a':
        return [dict(threshold=t, beta=b) for t, b in itertools.product(range(6), betas)]
    if preset == 'fig5b':
        return [dict(threshold=t, beta=b) for t, b in itertools.product(range(1, 5), fine_betas)]
    rs = [x / 10 for x in range(29, 51)]
    if preset == 'fig6a':
        return [dict(r=r, beta=b, pressure=p) for r, b, p in
                itertools.product(rs, [0.1, 0.2], ['dependent', 'independent'])]
    if preset == 'fig6b':
        return [dict(r=r, state=s) for r, s in itertools.product(rs, ['joint', 'behavior', 'environment'])]
    raise ValueError(preset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preset', choices=['fig2a', 'fig2b', 'fig3-4', 'fig5a', 'fig5b', 'fig6a', 'fig6b'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--mcs', type=int, default=1000000)
    parser.add_argument('--tail', type=int, default=100000)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.tail <= args.mcs:
        parser.error('Invalid repeat count or averaging window')
    configs = []
    for parameters in grid(args.preset):
        for repeat in range(args.repeats):
            configs.append(dict(parameters, mcs=args.mcs, tail=args.tail, seed=args.seed + len(configs)))
    with Path(args.output).open('x', encoding='utf8') as f:
        json.dump(configs, f, indent=2)
    print(f'Wrote {len(configs)} independent runs to {args.output}; no simulations started.')


if __name__ == '__main__':
    main()
