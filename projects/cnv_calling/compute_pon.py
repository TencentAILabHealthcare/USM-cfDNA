# Copyright (c) 2025, Tencent Inc. All rights reserved.
import sys

sys.path.append(".")

import time
import os
import argparse
import json
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from tgnn.sci.parser.wig_parsing import parse_wig
from read_counter import read_counter
from tgnn.sci.constants.chr_constants import CHROM_ARMS


def parse_wig_to_dataframe(filename):
    data_points = []
    with open(filename, 'r') as f:
        for chrom, start, end, value in parse_wig(f):
            if chrom not in CHROM_ARMS:
                continue

            region = CHROM_ARMS[chrom]
            if end <= region['p'][1]:
                arm = 'p'
            else:
                arm = 'q' if start >= region['q'][0] else None
            arm = chrom + arm if arm is not None else None
            data_points.append((chrom, start, end, arm, float(value)))
    print(f"#bins in {os.path.basename(filename)}:", len(data_points))
    df = pd.DataFrame(data_points, columns=['chrom', 'start', 'end', 'arm', 'reads'])
    return df


def parse_bam_to_dataframe(filename,
                           chrs=None,
                           window=1000,
                           min_mapq=30,
                           num_workers=None):
    print(f"parsing read count form alignment file: {filename}")
    counters = read_counter(filename,
                            chrs=chrs,
                            window=window,
                            min_mapq=min_mapq,
                            num_workers=num_workers)
    data_points = []
    for counter in counters:
        chrom = counter.chrom
        if chrom not in CHROM_ARMS:
            continue

        region = CHROM_ARMS[chrom]
        for i in range(len(counter)):
            start, end, value = counter[i]
            if end <= region['p'][1]:
                arm = 'p'
            else:
                arm = 'q' if start >= region['q'][0] else None

            arm = chrom + arm if arm is not None else None
            data_points.append((chrom, start, end, arm, value))

    print(f"#bins in {os.path.basename(filename)}:", len(data_points))
    df = pd.DataFrame(data_points, columns=['chrom', 'start', 'end', 'arm', 'reads'])
    return df


def gc_correction(x, mappability=0.75):
    assert {"reads", "gc", "map"} <= set(x.columns), "Missing one of required columns: reads, gc, map"
    # Initialize valid and ideal flags
    x['valid'] = True
    x.loc[np.array(x['reads'] <= 0) | np.array(x['gc'] < 0), 'valid'] = False
    x['ideal'] = True
    # Apply the ideal condition based on the thresholds
    x.loc[~x['valid'] | (x['map'] < mappability), 'ideal'] = False
    # Perform LOESS fitting
    z_predict = lowess(x.loc[x['ideal'], 'reads'], x.loc[x['ideal'], 'gc'], frac=0.5, return_sorted=False)
    # Normalize by the median of the predictions
    z_scale = z_predict / np.median(z_predict)
    x.loc[x['ideal'], 'gc_corrected_reads'] = x.loc[x['ideal'], 'reads'] / z_scale

    return x


def compute_pon(
        input_files,
        gc_file,
        map_file,
        mappability_threshold=0.75,
        chrs=None,
        window=1000_000,
        min_mapq=30,
        num_workers=None
):
    """
    Compute the Panel of Normals (PoN) from multiple normal sample WIG files.

    Args:
        input_files: List of paths to input WIG files for normal samples.
        gc_file: Path to the GC content file.
        map_file: Path to the mappability file.
    """
    map_file = parse_wig_to_dataframe(map_file)
    gc_file = parse_wig_to_dataframe(gc_file)
    gc_file.loc[gc_file['reads'] < 0, 'reads'] = 0
    # Read and combine all normal sample WIG files
    all_chrom_bin = []
    all_chrom_arm = []
    for file_path in input_files:
        if file_path.endswith('.wig'):
            df = parse_wig_to_dataframe(file_path)
        elif file_path.endswith(("bam", "sam", "cram")):
            df = parse_bam_to_dataframe(file_path,
                                        chrs=chrs,
                                        window=window,
                                        min_mapq=min_mapq,
                                        num_workers=num_workers)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        df['map'] = map_file['reads']
        df['gc'] = gc_file['reads']
        gc_correction(df, mappability=mappability_threshold)

        df.index = list(
            map(lambda x: f'{x[0]}:{x[1]}-{x[2]}', df.loc[:, ['chrom', 'start', 'end']].values))

        result_bin = df[df['ideal']].loc[:, ['gc_corrected_reads']].T
        result_bin.index.name = None
        result_bin.index = [0]
        all_chrom_bin.append(result_bin)

        result_arm = df[~df['arm'].isna()].groupby("arm").agg({
            'gc_corrected_reads': lambda arr: sum(list(filter(lambda x: not np.isnan(x) and x > 0, arr)))
        }).T
        result_arm.index.name = None
        result_arm.index = [0]
        all_chrom_arm.append(result_arm)

    all_chrom_arm = pd.concat(all_chrom_arm, ignore_index=True)
    all_chrom_bin = pd.concat(all_chrom_bin, ignore_index=True)
    all_chrom_arm_perc = pd.DataFrame(
        all_chrom_arm.to_numpy().T / np.array(all_chrom_arm.sum(1)),
        index=all_chrom_arm.columns,
        columns=all_chrom_arm.index
    ).T.fillna(0) * 100 + 1e-10

    all_chrom_arm_perc = all_chrom_arm_perc.loc[:, sorted(all_chrom_arm_perc.columns, key=lambda x: int(x[3:-1]))]
    all_chrom_bin_perc = pd.DataFrame(
        all_chrom_bin.to_numpy().T / np.array(all_chrom_bin.sum(1)),
        index=all_chrom_bin.columns,
        columns=all_chrom_bin.index
    ).T.fillna(0) * 100 + 1e-10

    all_chrom_bin_perc = all_chrom_bin_perc.loc[:, sorted(all_chrom_bin_perc.columns,
                                                          key=lambda x: (int(x.split(':')[0][3:]),
                                                                         int(x.split('-')[0].split(':')[1])))]
    return {
        'arm': {
            'mean': all_chrom_arm_perc.mean(0).to_dict(),
            'std': all_chrom_arm_perc.std(0).to_dict()
        },
        'bin': {
            'mean': all_chrom_bin_perc.mean(0).to_dict(),
            'std': all_chrom_bin_perc.std(0).to_dict()
        }
    }


def main(args):
    if args.window.isdigit():
        window = int(args.window)
    elif args.window.lower().endswith("kb"):
        window = int(args.window[:-2]) * 1000
    elif args.window.lower().endswith("mb"):
        window = int(args.window[:-2]) * 1000000
    else:
        raise ValueError(f"{args.window} is not a valid window type")
    assert window > 0, f"window must be greater than 0"
    ref_dir = os.path.dirname(os.path.abspath(__file__)) + "/reference"
    gc_ref_file = f"{ref_dir}/gc_hg38_{args.window}.wig"
    map_ref_file = f"{ref_dir}/map_hg38_{args.window}.wig"
    norm_files = set(args.input)
    assert len(norm_files) > 0, f"no normal files in: {norm_files}"
    for path in norm_files:
        assert os.path.exists(path), f"norm file does not exist: {path}"

    num_workers = args.num_threads if args.num_threads is None else int(args.num_threads)
    threshold = args.mappability_threshold
    print(f"mappability threshold: {threshold}")
    start = time.time()
    output = compute_pon(
        norm_files,
        gc_ref_file,
        map_ref_file,
        mappability_threshold=threshold,
        chrs=args.chrs,
        window=window,
        min_mapq=args.min_mapq,
        num_workers=num_workers
    )
    end = time.time()
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Finish Computing Panel of Normals (PoN). \nSaved Output to {args.output}")
    print(f"Time taken: {end - start:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
            create panel of normals for CNA computation 
            Examples: 
                >>> python3 compute_pon.py --input normal_1.wig normal_2.wig ... normal_n.wig
                >>> python3 compute_pon.py --input normal_1.bam normal_2.bam ... normal_n.bam
            """
    )
    parser.add_argument('--input', '-i', type=str, nargs='+', required=True,
                        help='Input WIG files (space separated) containing read depth information for normal samples.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory to save the computed panel of normals (PoN).')
    parser.add_argument('--mappability_threshold', '-mt', type=float, default=0.75,
                        help='Mappability threshold for filtering bins (default: 0.75).')
    parser.add_argument("--chrs", "-c", nargs='+', default=None, help="chromosomes for processing")
    parser.add_argument('-w', '--window', default="1000kb", help='window size (default: 1MB)')
    parser.add_argument('-q', '--min_mapq', type=int, default=30, help='minimum mapping quality (default: 30)')
    parser.add_argument('-t', '--num_threads', type=int, default=None, help='number of threads')
    args = parser.parse_args()
    main(args)
