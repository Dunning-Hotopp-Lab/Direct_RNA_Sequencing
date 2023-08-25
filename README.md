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
                   [--min_region_positional_depth MIN_REGION_POSITIONAL_DEPTH] [--candidate_offset_len CANDIDATE_OFFSET_LEN]
                   [--depth_delta_threshold DEPTH_DELTA_THRESHOLD] [--assigned_read_ratio_threshold ASSIGNED_READ_RATIO_THRESHOLD]
                   [-o OUTPUT_PREFIX] [--candidates_only] [-v]

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
  --min_region_positional_depth MIN_REGION_DEPTH
                        Minimum required positional depth within a region. Reads mapping to positions that are below this threshold are tossed
                        and new distinct subregions are created from the remaining runs of positions.
  --candidate_offset_len CANDIDATE_OFFSET_LEN
                        Some of the operon transcript candidates may have very close start or end coordinates. When encountering a start or end
                        coordinate, ensure the next candidate position is at minimum <offset_length> bases away.
  --depth_delta_threshold DEPTH_DELTA_THRESHOLD
                        If the absolute change of depth between two consecutive positions exceeds this number, consider this a potential start
                        or stop site.
  --candidate_read_threshold CANDIDATE_READ_THRESHOLD
                        If the number of reads overlapping within a candidate transcript is below this threshold, throw out the transcript.
  --assigned_read_ratio_threshold ASSIGNED_READ_RATIO_THRESHOLD
                        The ratio of reads assigned solely to a candidate transcript to all reads overlapping within the same transcript. Keep
                        transcripts that exceed this ratio.
  -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
                        Prefix of the output files to save results to. The script will append .bed and .gff3 to the filenames
  --candidates_only     If enabled, skip the candidate transcript-filtering step and print out the candidate transcripts only
  -v, --verbose         If enabled, print detailed step progress.

--reads_bed file uses the first 6 standard BED columns (chr, start, end, name, score, strand)
--ranges_bed uses the first 4 standard BED columns (chr, start, end, strand)

The first output file ('.candidates.bed') is a 6-column BED file of all candidate transcripts for each chromsome. The 6 columns are (chr, start, end, name, score, strand).The second output file is a 6-column BED file of all predicted transcripts for each chromsome that passed our linear model fitting and prediction cutoffs. The columns are the same as the first file.The third output file is a 9-column GFF3 file of the same contents from the second BED file. The 'score' column is the normalized score from predicting the best transcripts for a region.

```
