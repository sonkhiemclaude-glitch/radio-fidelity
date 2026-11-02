"""Paired whole-spatial-block bootstrap of saved, fixed-model median errors.

Same block draws for all methods and seeds. Seed medians are averaged, matching
the tables; no seed-population or new-deployment confidence claim is made.
"""
import argparse, hashlib, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import dataset as D
ROOT = Path(__file__).resolve().parents[1]
B = 2000
SEED = 20261102

def block_medians(error, inverse, multiplicities, batch=64):
    """Exact ordinary medians of replicated whole blocks, without replication."""
    error = np.asarray(error)
    order = np.argsort(error, kind='stable')
    sorted_error = error[order]
    block_order = np.asarray(inverse)[order]
    answer = []
    for start in range(0, len(multiplicities), batch):
        counts = multiplicities[start:start+batch, block_order]
        cumulative = np.cumsum(counts, axis=1)
        n = cumulative[:, -1]
        if np.any(n <= 0):
            raise ValueError('Empty resample')
        lo = (n-1)//2+1
        hi = n//2+1
        low_index = (cumulative >= lo[:, None]).argmax(axis=1)
        high_index = (cumulative >= hi[:, None]).argmax(axis=1)
        answer.append((sorted_error[low_index]+sorted_error[high_index])/2)
    return np.concatenate(answer)

def old(arm, mode, level, kind):
    return [f'pred_{arm}_{mode}_F{level}_s{s}_{kind}.npz' for s in range(3)]

def mapped(tag, mode, level=4):
    return [f'pred_mapcnn_{tag}_{mode}_s{s}.npz' if mode == 'real'
            else f'pred_mapcnn_{tag}_{mode}_F{level}_s{s}.npz' for s in range(3)]

def comparisons(include_map):
    rows = []
    def add(name, a, b):
        rows.append(dict(comparison=name, a=a, b=b))
    real_knn = [f'pred_real_s{s}_knn.npz' for s in range(3)]
    for kind in ['mlp', 'transformer']:
        add(f'real_{kind}_minus_knn', [f'pred_real_s{s}_{kind}.npz' for s in range(3)], real_knn)
    for arm in ['free', 'equal']:
        for mode in ['guess', 'oracle']:
            for kind in ['knn', 'mlp', 'transformer']:
                add(f'{arm}_{mode}_{kind}_F4_minus_F5', old(arm, mode, 4, kind), old(arm, mode, 5, kind))
            add(f'{arm}_{mode}_F4_transformer_minus_knn', old(arm, mode, 4, 'transformer'), old(arm, mode, 4, 'knn'))
        for kind in ['knn', 'mlp', 'transformer']:
            add(f'{arm}_{kind}_F4_fitted_minus_prior', old(arm, 'oracle', 4, kind), old(arm, 'guess', 4, kind))
    if include_map:
        for mode in ['guess', 'oracle']:
            add(f'equal_{mode}_F4_map_minus_no_map', mapped('map', mode), mapped('no_map', mode))
            add(f'equal_{mode}_F4_map_minus_transformer', mapped('map', mode), old('equal', mode, 4, 'transformer'))
            for tag in ['map', 'no_map']:
                add(f'equal_{mode}_{tag}_F4_minus_F5', mapped(tag, mode, 4), mapped(tag, mode, 5))
        for tag in ['map', 'no_map']:
            add(f'equal_{tag}_F4_fitted_minus_prior', mapped(tag, 'oracle'), mapped(tag, 'guess'))
            add(f'real_{tag}_minus_knn', mapped(tag, 'real'), real_knn)
        add('real_map_minus_no_map', mapped('map', 'real'), mapped('no_map', 'real'))
    return rows

def compute(job):
    name, block_m, inverse, multiplicities, query = job
    z = np.load(ROOT/'data'/name)
    if not np.array_equal(z['query'], query):
        raise ValueError('Mismatched query order: '+name)
    error = z['error']
    if not np.all(np.isfinite(error)) or np.any(error < 0):
        raise ValueError('Invalid errors: '+name)
    # Cache includes all inputs, so changed predictions invalidate it.
    digest = hashlib.sha256()
    for item in [error, inverse, multiplicities]:
        digest.update(item.tobytes())
    key = digest.hexdigest()
    cache = ROOT/'results/intermediate/bootstrap'
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f'{block_m}_{key}.npz'
    if target.exists():
        draws = np.load(target)['median']
    else:
        draws = block_medians(error, inverse, multiplicities)
        # Unique temporary path tolerates identical deterministic kNN seeds.
        import os
        tmp = target.with_suffix(f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(tmp, median=draws)
        tmp.replace(target)
    return name, float(np.median(error)), draws

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    start = time.time()
    specs = comparisons(not args.existing_only)
    names = sorted({name for row in specs for key in ['a', 'b'] for name in row[key]})
    query = np.load(ROOT/'data'/names[0])['query']
    d = D.load()
    xy = np.c_[d['x'][query], d['y'][query]]
    out = dict(protocol=dict(B=B, seed=SEED, block_sizes_m=[500, 1000], seeds_held_fixed=3,
              n_query=len(query), query_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
              statistic='mean across three seed medians for A minus that for B; negative favors A',
              sampling='whole spatial blocks drawn uniformly with replacement; paired across methods and seeds',
              uncertainty='pointwise percentile 95% intervals, no multiplicity correction; conditional on trained models and spatial-bootstrap approximation'), rows=[])
    replicate_output = {}
    for block_m in [500, 1000]:
        blocks, inverse = np.unique(np.floor(xy/block_m).astype(int), axis=0, return_inverse=True)
        count = len(blocks)
        multiplicities = np.random.default_rng(SEED+block_m).multinomial(count, np.full(count, 1/count), size=B)
        jobs = [(name, block_m, inverse, multiplicities, query) for name in names]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = {name:(point, boot) for name, point, boot in pool.map(compute, jobs)}
        for spec in specs:
            ap = np.array([results[n][0] for n in spec['a']])
            bp = np.array([results[n][0] for n in spec['b']])
            boot = np.mean([results[n][1] for n in spec['a']], axis=0)-np.mean([results[n][1] for n in spec['b']], axis=0)
            row = dict(**spec, block_m=block_m, n_blocks=count, difference_m=float(np.mean(ap-bp)),
                       seed_differences_m=(ap-bp).tolist(), ci95_m=np.percentile(boot,[2.5,97.5]).tolist())
            out['rows'].append(row)
            replicate_output[f"{block_m}_{spec['comparison']}"] = boot
            print({k:v for k,v in row.items() if k not in ['a','b','seed_differences_m']}, flush=True)
    out['runtime_s'] = time.time()-start
    stem = 'exp18_existing_bootstrap' if args.existing_only else 'exp18_spatial_bootstrap'
    np.savez_compressed(ROOT/'data'/f'{stem}_replicates.npz', **replicate_output)
    (ROOT/'results'/f'{stem}.json').write_text(json.dumps(out, indent=2))
    print('COMPLETE',len(out['rows']),out['runtime_s'],flush=True)

if __name__ == '__main__':
    main()
