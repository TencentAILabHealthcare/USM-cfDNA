# Copyright (c) 2024, Tencent Inc. All rights reserved.

import os
from collections import OrderedDict
from typing import Tuple, Sequence, Union, Dict
from Bio import Seq
from Bio.SeqIO import FastaIO, SeqRecord

def parse_fasta(fasta_string: str, to_dict=False) -> Union[Tuple[Sequence[str], Sequence[str], Sequence[str]], Dict]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Args:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence ids
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    ids = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            seq_id, *description = line[1:].split(None, 1)  # Remove the '>' at the beginning.
            ids.append(seq_id)
            if len(description) > 0:
                descriptions.append(description)
            else:
                descriptions.append("")
            sequences.append('')
            continue
        elif line.startswith('#'):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    if to_dict:
        return OrderedDict(zip(ids, sequences))

    return sequences, ids, descriptions


def export_fasta(sequences, ids=None, descriptions=None, output=None):
    fh = None
    if output is not None:
        os.makedirs(os.path.dirname(os.path.realpath(output)), exist_ok=True)
        fh = open(output, "w")

    if ids is None:
        ids = [f"sequence{i}" for i in range(len(sequences))]

    if descriptions is None:
        descriptions = ["" for i in range(len(sequences))]

    fasta_string = []
    for seq, seq_id, desc in zip(sequences, ids, descriptions):
        fstring = FastaIO.as_fasta(SeqRecord(Seq.Seq(seq),
                                             id=seq_id,
                                             description=desc))
        if fh is not None:
            fstring.write(fh)
        else:
            fasta_string.append(fstring)

    if fh is not None:
        fh.close()
        return output
    else:
        return "".join(fasta_string)
