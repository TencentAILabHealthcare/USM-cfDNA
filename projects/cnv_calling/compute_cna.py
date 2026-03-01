# Copyright (c) 2025, Tencent Inc. All rights reserved.
import sys
import time

sys.path.append(".")

import argparse
import pandas as pd
import numpy as np
import json
import os

from compute_pon import parse_wig_to_dataframe, gc_correction, compute_pon, parse_bam_to_dataframe
from viz_cna import visualize_cna


def compute_cna(
        input_file,
        output_file,
        gc_file,
        map_file,
        pon_file,
        mappability_threshold=0.75,
        window=1000_000,
        chrs=None,
        min_mapq=30,
        num_workers=None
):
    """
    Compute Copy Number Alterations (CNA) for a sample using the provided PoN.

    Args:
        input_file: Path to the input WIG or BAM file for the sample.
        pon_file: Path to the PoN file.
        output_file: Path to save the computed CNA results.
        gc_file: Path to the GC content file.
        map_file: Path to the mappability file.
    """
    assert os.path.exists(pon_file), f"pon file {pon_file} does not exist."
    if input_file.endswith('.bam'):
        df = parse_bam_to_dataframe(input_file,
                                    window=window,
                                    chrs=chrs,
                                    min_mapq=min_mapq,
                                    num_workers=num_workers)
    else:
        df = parse_wig_to_dataframe(input_file)

    df['map'] = parse_wig_to_dataframe(map_file)['reads']
    df['gc'] = parse_wig_to_dataframe(gc_file)['reads']
    df.loc[df['gc'] < 0, 'gc'] = 0
    gc_correction(df, mappability=mappability_threshold)

    df.index = list(
        map(lambda x: f'{x[0]}:{x[1]}-{x[2]}', df[['chrom', 'start', 'end']].values))

    result_bin = df[df['ideal']].loc[:, ['gc_corrected_reads']].T
    result_bin.index.name = None
    result_bin.index = [0]

    result_arm = df[~df['arm'].isna()].groupby("arm").agg({
        'gc_corrected_reads': lambda arr: sum(list(filter(lambda x: not np.isnan(x) and x > 0, arr)))
    }).T

    result_arm.index.name = None
    result_arm.index = [0]
    result_arm_perc = pd.DataFrame(result_arm.to_numpy().T / np.array(result_arm.sum(1)), index=result_arm.columns,
                                   columns=result_arm.index).T.fillna(0) * 100 + 1e-10
    result_arm_perc = result_arm_perc.loc[:, sorted(result_arm_perc.columns, key=lambda x: int(x[3:-1]))]

    result_bin_perc = pd.DataFrame(result_bin.to_numpy().T / np.array(result_bin.sum(1)), index=result_bin.columns,
                                   columns=result_bin.index).T.fillna(0) * 100 + 1e-10
    result_bin_perc = result_bin_perc.loc[
        :, sorted(result_bin_perc.columns,
                  key=lambda x: (int(x.split(':')[0][3:]), int(x.split('-')[0].split(':')[1])))]

    print(f"loading pon file: {pon_file}")
    with open(pon_file, 'r') as f:
        pon = json.load(f)
    cna_arm = (result_arm_perc - pd.Series(pon['arm']['mean'])) / pd.Series(pon['arm']['std'])
    cna_bin = (result_bin_perc - pd.Series(pon['bin']['mean'])) / pd.Series(pon['bin']['std'])

    if output_file.endswith(".json"):
        cna = {
            'arm': cna_arm.to_dict(orient='records')[0],
            'bin': cna_bin.to_dict(orient='records')[0]
        }
        with open(output_file, 'w') as f:
            json.dump(cna, f, indent=4)

    print(f"CNA results saved to {output_file}")


def main(args):
    ref_dir = os.path.dirname(os.path.abspath(__file__)) + "/reference"
    gc_ref_file = f"{ref_dir}/gc_hg38_{args.window}.wig"
    map_ref_file = f"{ref_dir}/map_hg38_{args.window}.wig"
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    if args.window.isdigit():
        window = int(args.window)
    elif args.window.lower().endswith("kb"):
        window = int(args.window[:-2]) * 1000
    elif args.window.lower().endswith("mb"):
        window = int(args.window[:-2]) * 1000000
    else:
        raise ValueError(f"{args.window} is not a valid window type")
    assert window > 0, f"window must be greater than 0"

    num_workers = args.num_threads if args.num_threads is None else int(args.num_threads)
    pon_file = args.pon_file
    if args.chrs is None:
        chrs = [f"chr{i}" for i in range(1, 23)]
    else:
        chrs = args.chrs

    if args.pon_file is None:
        assert args.norm_files is not None, f"not exist norm_files:{args.norm_files}, pon file or normal files must be provided"
        print(f"compute pon files: {args.norm_files}")
        norm_files = set(args.norm_files)
        assert len(norm_files) > 0, f"no normal files in: {norm_files}"
        for path in norm_files:
            assert os.path.exists(path), f"norm file does not exist: {path}"
        output = compute_pon(norm_files,
                             gc_ref_file,
                             map_ref_file,
                             mappability_threshold=args.mappability_threshold,
                             chrs=chrs,
                             window=window,
                             min_mapq=args.min_mapq,
                             num_workers=num_workers)
        name = os.path.basename(args.output).split(".")[0]
        pon_file = f"{out_dir}/{name}_pon.json"
        with open(pon_file, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Finish Computing Panel of Normals (PoN). \nSaved Output to {args.output}")
    assert os.path.exists(pon_file), f"pon file does not exist: {pon_file}"
    print(f"Input PoN File: {pon_file}")
    start = time.time()
    compute_cna(args.input,
                args.output,
                pon_file=pon_file,
                gc_file=gc_ref_file,
                chrs=chrs,
                mappability_threshold=args.mappability_threshold,
                window=window,
                map_file=map_ref_file,
                num_workers=num_workers)
    end = time.time()
    print(f"Finish Computint CNV. \nSaved Output to {args.output}")
    print(f"Time taken: {end - start:.2f} seconds")

    if args.visualize:
        name = os.path.basename(args.output).split(".")[0]
        visualize_cna(args.output, f"{out_dir}/{name}.png", pon_file=pon_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate WIG file from BAM read coverage.')
    parser.add_argument('-i', '--input', required=True, help='Input bam or wig file')
    parser.add_argument('-o', '--output', help='Path to output')
    parser.add_argument('-n', '--norm_files', default=None, help='Input bam or wig file list for normalization')
    parser.add_argument('-p', '--pon_file', default=None, help='Input PoN File')
    parser.add_argument('--mappability_threshold', '-mt', type=float, default=0.75,
                        help='Mappability threshold for filtering bins (default: 0.75).')
    parser.add_argument("--chrs", "-c", nargs='+', default=None, help="Chromosomes for processing")
    parser.add_argument('-w', '--window', type=str, default="1000kb", help='Window size (default: 1000kb)')
    parser.add_argument('-q', '--min_mapq', type=int, default=30, help='Minimum mapping quality (default: 30)')
    parser.add_argument("-v", '--visualize', action="store_true", help="visualize cnv results")
    parser.add_argument('-t', '--num_threads', type=int, default=None, help='number of threads of computing PoN')
    args = parser.parse_args()
    main(args)
