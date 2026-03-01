# Copyright (c) 2024, Tencent Inc. All rights reserved.

iupac_base_types = (
    "A",
    "C",
    "G",
    "T",
    "U",
    "R",
    "Y",
    "S",
    "W",
    "K",
    "M",
    "B",
    "D",
    "H",
    "V",
    "N"
)

iupac_to_base = {
    "A": ["A", ],
    "C": ["C", ],
    "G": ["G", ],
    "T": ["T", ],
    "U": ["U", ],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
    "N": ["N", ]
}

aa_to_codon = {
    '*': ['TAA', 'TAG', 'TGA'],  # Stop.
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],  # Ala.
    'C': ['TGT', 'TGC'],  # Cys.
    'D': ['GAT', 'GAC'],  # Asp.
    'E': ['GAA', 'GAG'],  # Glu.
    'F': ['TTT', 'TTC'],  # Phe.
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],  # Gly.
    'H': ['CAT', 'CAC'],  # His.
    'I': ['ATT', 'ATC', 'ATA'],  # Ile.
    'K': ['AAA', 'AAG'],  # Lys.
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],  # Leu.
    'M': ['ATG'],  # Met.
    'N': ['AAT', 'AAC'],  # Asn.
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],  # Pro.
    'Q': ['CAA', 'CAG'],  # Gln.
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arg.
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],  # Ser.
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],  # Thr.
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],  # Val.
    'W': ['TGG'],  # Trp.
    'Y': ['TAT', 'TAC'],  # Tyr.
}

def iupac_to_acgt(base):
    return iupac_to_base[base][0]

genotypes = ("0/0", "1/1", "0/1", "1/2")
base_types = ("A", "C", "G", "T", "U", "N")
base14_types = ("A", "C", "G", "T", "N", "DS", "IS", "a", "c", "g", "t", "n", "ds", "is")
base24_types = (
    "A", "C", "G", "T", "N", "D", "I", "a", "c", "g", "t", "n", "d", "i", "A+", "C+", "G+", "T+", "N+", "a+", "c+",
    "g+", "t+", "n+")

base38_types = (
    "A", "C", "G", "T", "N", "a", "c", "g", "t", "n",
    "A+", "C+", "G+", "T+", "N+", "a+", "c+", "g+", "t+", "n+",
    "A-", "C-", "G-", "T-", "N-", "a-", "c-", "g-", "t-", "n-",
    "DS", "DE", "IS", "IE", "ds", "de", "is", "ie"
)

# acgtn is insertion base, '-': deletion, '.':gap
alt_base_types = ("A", "C", "G", "T", "N", "a", "c", "g", "t", "n", "-", ".")