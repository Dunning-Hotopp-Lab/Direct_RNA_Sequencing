# Direct_RNA_Sequencing

## RDOperon

Python 3.8 and later are supported. This was tested using Python 3.11.3

### Installation

#### Route 1 - pip

You can install the required dependencies by running `python3 -m pip install -r requirements.txt`

#### Route 2 - Docker

You can also use the provided Dockerfile to build a Docker image to run the tool in. To build the Docker image, run `docker build -t rdoperon .`

### Running the tool

The general command is `python3 rdoperon.py <options>`

If you are using Docker to run the tool then the command would instead be `docker run -it  -v "$PWD":/usr/src/app rdoperon <options>`

```text
usage: rdoperon.py [-h] -r READS_BED -d RANGES_BED [-c CHROMOSOME] [--min_depth MIN_DEPTH] [--max_depth MAX_DEPTH]
                   [--min_region_depth MIN_REGION_DEPTH] [--bin_size BIN_SIZE] [--depth_delta_threshold DEPTH_DELTA_THRESHOLD]
                   [--model_assigned_read_threshold MODEL_ASSIGNED_READ_THRESHOLD] [-o OUTPUT_PREFIX] [-v]

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
  --min_region_depth MIN_REGION_DEPTH
                        Minimum required positional depth within a region. Reads mapping to positions that are below this threshold are tossed and new distinct
                        subregions are created from the remaining runs of positions.
  --bin_size BIN_SIZE   Some of the operon transcript candidates may have very close start or end coordinates. Perform binning of the specified size on the entire
                        transcript region and take the min start or max end positions per bin amongst the candidate positions.
  --depth_delta_threshold DEPTH_DELTA_THRESHOLD
                        If the absolute change of depth between two consecutive positions exceeds this number, consider this a potential start or stop site.
  --model_assigned_read_threshold MODEL_ASSIGNED_READ_THRESHOLD
                        When assigning reads to candidate transcripts (models), throw out any models below this threshold of reads assigned to it.
  -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
                        Prefix of the output files to save results to. The script will append .bed and .gff3 to the filenames
  -v, --verbose         If enabled, print detailed step progress.

--reads_bed file uses the first 6 standard BED columns (chr, start, end, name, score, strand)
--ranges_bed uses the first 4 standard BED columns (chr, start, end, strand)

The first output file ('.candidates.bed') is a 6-column BED file of all candidate transcripts for each chromsome. The 6 columns are (chr, start, end, name, score, strand).
The second output file is a 6-column BED file of all predicted transcripts for each chromsome that passed our linear model fitting and prediction cutoffs. The columns are the same as the first file.
The third output file is a 9-column GFF3 file of the same contents from the BED file. The 'score' column is the normalized score from predicting the best transcripts for a region.

```
