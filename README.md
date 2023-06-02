# Direct_RNA_Sequencing

## RDOperon

Currently tested with Python 3.11.3

You can install the required dependencies by running `python3 -m pip install -r requirements.txt`

```text
usage: rdoperon.py [-h] -r READS_BED -d RANGES_BED [-c CHROMOSOME] [-b NUM_BINS]

Predict operons based on a finding candidate start/stop sites, and filter alternative transcripts that could make use of various start/stop combinations.

options:
  -h, --help            show this help message and exit
  -r READS_BED, --reads_bed READS_BED
                        Path to a position-sorted BED file of reads.
  -d RANGES_BED, --ranges_bed RANGES_BED
                        Path to a BED file of distict non-overlapping interval ranges per strand
  -c CHROMOSOME, --chromosome CHROMOSOME
                        Specific check on a chromsome ID. If not provided, will loop over all chromosomes
  -b NUM_BINS, --num_bins NUM_BINS
                        Set number of total bins per region.

--reads_bed file uses the first 6 standard BED columns (chr, start, end, name, score, strand)
--ranges_bed uses the first 4 standard BED columns (chr, start, end, strand)

Output file is a 6-column BED file of all candidate alternative transcripts for each chromsome. The 6 columns are (chr, start, end, name, score, strand)
```