# Copyright (c) 2025, Tencent Inc. All rights reserved.

import bisect
import io
import sys
from collections import defaultdict

import math
from typing import Generator, Tuple, Union


def parse_header(line):
    header = {}
    splits = line.split()
    for item in splits[1:]:
        try:
            key, value = item.split('=')
            if key in ("start", "step", "span"):
                value = int(value)

            header[key] = value
        except(KeyError, IndexError, ValueError):
            print(f"WARNING: can't nor parse key=value in header: {item}", file=sys.stderr)
            continue
    # 0-based
    if "start" in header:
        header["start"] = header["start"] - 1

    return header


def parse_wig(wig_file: Union[str, io.IOBase]) -> Generator[Tuple[str, int, int, float], None, None]:
    current_block = None
    current_params = {}
    if isinstance(wig_file, io.IOBase):
        f = wig_file
    else:
        f = wig_file.splitlines()

    for i, line in enumerate(f):
        if not line or line.startswith(('track', '#')):
            continue

        # fix step or variable step
        if line.startswith(('variableStep', 'fixedStep')):
            current_block = 'fixed' if line.startswith('fixed') else 'variable'
            current_params = parse_header(line.strip())
            if 'chrom' not in current_params:
                raise ValueError(f"Missing 'chrom' in header at line {i}")

            if 'span' not in current_params:
                current_params['span'] = 1

            continue

        # process data line
        if not current_block:
            raise RuntimeError(f"Data without header at line {i}")

        line = line.strip()
        if current_block == 'fixed':
            try:
                value = float(line)
            except ValueError:
                raise ValueError(f"Invalid value at line {i}: {line}")

            # already 0-based coord
            start = current_params['start']
            step = current_params['step']
            span = current_params['span']
            # start and end is python style
            end = start + span
            yield (current_params['chrom'], start, end, value)
            # update start position
            current_params['start'] = start + step
        else:
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid data format at line {i}: {line}")

            try:
                # to 0-based coord
                start = int(parts[0]) - 1
                value = float(parts[1])
            except ValueError:
                raise ValueError(f"Invalid value at line {i}: {line}")

            span = current_params['span']
            # start and end is python style
            end = start + span
            yield (current_params['chrom'], start, end, value)


class FixStepCounter:
    def __init__(self, chrom, window, start=0, end=None):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.window = window
        self.counts = defaultdict(int)

    def update(self, position, n=1):
        if self.end is not None:
            assert self.start <= position < self.end, f"{position} is out of range"
        self.counts[(position - self.start) // self.window] += n

    def __len__(self):
        return max(self.counts.keys()) + 1

    def __getitem__(self, index):
        if index >= max(self.counts.keys()) + 1:
            raise StopIteration

        start = self.start + index * self.window
        end = start + self.window
        return start, end, self.counts[index]

    @classmethod
    def from_string(cls, wig_string):
        counters = {}
        current_params = None
        for i, line in enumerate(wig_string.splitlines()):
            if not line or line.startswith(('track', '#')):
                continue

            # fix step or variable step
            if line.startswith('fixedStep'):
                current_params = parse_header(line.strip())
                if 'chrom' not in current_params:
                    raise ValueError(f"Missing 'chrom' in header at line {i}")
                counters[current_params['chrom']] = cls(current_params['chrom'],
                                                        current_params['step'],
                                                        current_params['start'])
                if 'span' not in current_params:
                    current_params['span'] = 1

                continue

            if not current_params:
                raise RuntimeError(f"Data without header at line {i}")

            line = line.strip()
            try:
                value = float(line)
            except ValueError:
                raise ValueError(f"Invalid value at line {i}: {line}")

            # already 0-based coord
            start = current_params['start']
            step = current_params['step']
            counters[current_params['chrom']].update(start, n=value)

            # update start position
            current_params['start'] = start + step

        return counters

    def to_string(self):
        # 1-based coord
        lines = [f'fixedStep chrom={self.chrom} start={self.start + 1} step={self.window} span={self.window}']
        for i in range(len(self)):
            lines.append(f"{int(self.counts[i])}")
        return "\n".join(lines)


class VariableStepCounter:
    def __init__(self, chrom, starts, span=1):
        self.chrom = chrom
        starts = sorted(starts)
        self.start = starts[0]
        self.end = starts[-1] + span
        self.counts = {start: 0 for start in starts}
        self.starts = starts
        self.span = span

    def update(self, position):
        assert self.start <= position < self.end, f"{position} is out of range"
        idx = bisect.bisect_right(self.starts, position)
        if self.starts[idx] + self.span < position:
            assert position < self.starts[idx] + self.span, f"{position} is out of range"
        self.counts[position] += 1

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        start = self.starts[index]
        end = start + self.span
        return start, end, self.counts[start]

    def to_string(self):
        lines = [f"variableStep chrom={self.chrom} span={self.span}"]
        # 1-based coord
        lines = lines + [f"{s + 1} {c}" for s, c in self.counts.items()]
        return "\n".join(lines)


def write_wig(wig_file, counters):
    counters = sorted(counters, key=lambda x: x.chrom)
    with open(wig_file, 'w') as f:
        f.write('track type=wiggle_0\n')
        for counter in counters:
            f.write(counter.to_string())
