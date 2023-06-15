# Direct_RNA_Sequencing

## RDOperon

Python 3.8 and later are supported. This was tested using Python 3.11.3


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
  --min_depth MIN_DEPTH
                        Minimum required depth for a distinct region to find candidate transcripts.
  --max_depth MAX_DEPTH
                        Maximum required depth for a distinct region to find candidate transcripts. If not set, max threshold is disabled.
  --num_stdev NUM_STDEV
                        Set the number of standard deviations to filter potential start and stop sites.
  -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
                        Prefix of the output files to save results to. The script will append .bed and .gff3 to the filenames
  -v, --verbose         If enabled, print detailed step progress.

--reads_bed file uses the first 6 standard BED columns (chr, start, end, name, score, strand)
--ranges_bed uses the first 4 standard BED columns (chr, start, end, strand)

The first output file is a 6-column BED file of all candidate alternative transcripts for each chromsome. The 6 columns are (chr, start, end, name, score, strand)The second output file is a 9-column GFF3 file of the same contents from the BED file. The 'score' column is the normalized score from predicting the best transcripts for a region.
```