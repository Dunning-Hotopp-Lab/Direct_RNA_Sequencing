# Direct_RNA_Sequencing

## TP (Transcript Prediction)

Python 3.9 and later are supported. This was tested using Python 3.11.3

This tool may appear under both `tp.py` (the script name) and `drs-transcript-prediction` (the executable name).

### Installation

#### Route 1 - pip

You can install the required dependencies by running `python3 -m pip install -r requirements.txt`

Alternatively, run `python3 -m pip install .` from the repository root to install the tool and its dependencies. This adds two equivalent commands to your PATH: `tp.py` and `drs-transcript-prediction`.

#### Route 2 - Docker

You can also use the provided Dockerfile to build a Docker image to run the tool in. To build the Docker image, run `docker build -t tp .`

### Running the tool

The general command is `python3 tp.py <options>`

If you installed the tool with `python3 -m pip install .`, you can instead run `tp.py <options>` or `drs-transcript-prediction <options>`.

If you are using Docker to run the tool then the command would instead be `docker run -it  -v "$PWD":/usr/src/app tp <options>`

```text
usage: tp.py [-h] -r READS_BED -d RANGES_BED [-c CHROMOSOME] [--min_depth MIN_DEPTH] [--max_depth MAX_DEPTH] [--min_region_positional_depth MIN_REGION_POSITIONAL_DEPTH]
             [--candidate_offset_len CANDIDATE_OFFSET_LEN] [--depth_delta_threshold DEPTH_DELTA_THRESHOLD] [--candidate_read_threshold CANDIDATE_READ_THRESHOLD]
             [--assigned_read_ratio_threshold ASSIGNED_READ_RATIO_THRESHOLD] [-o OUTPUT_PREFIX] [--candidates_only] [-v] [--version]

Predict transcripts based on a finding candidate start/stop sites, and filter alternative transcripts that could make use of various start/stop combinations.

options:
  -h, --help            show this help message and exit
  -r, --reads_bed READS_BED
                        (Required) Path to a position-sorted BED file of reads.
  -d, --ranges_bed RANGES_BED
                        (Required) Path to a BED file of distict non-overlapping interval ranges per strand
  -c, --chromosome CHROMOSOME
                        (Optional) Specific check on a chromsome ID. If not provided, will loop over all chromosomes
  --min_depth MIN_DEPTH
                        (Optional) Minimum required depth for a distinct region to find candidate transcripts. Default: 1
  --max_depth MAX_DEPTH
                        (Optional) Maximum required depth for a distinct region to find candidate transcripts. If not set, max threshold is disabled.
  --min_region_positional_depth MIN_REGION_POSITIONAL_DEPTH
                        (Optional) Minimum required positional depth within a region. Reads mapping to positions that are below this threshold are tossed and new distinct subregions are
                        created from the remaining runs of positions. Default: 2
  --candidate_offset_len CANDIDATE_OFFSET_LEN
                        (Optional) Some of the transcript candidates may have very close start or end coordinates. When encountering a start or end coordinate, ensure the next candidate
                        position is at minimum <offset_length> bases away. Default: 100
  --depth_delta_threshold DEPTH_DELTA_THRESHOLD
                        (Optional) If the absolute change of depth between two consecutive positions exceeds this number, consider this a potential start or stop site. Default: 4
  --candidate_read_threshold CANDIDATE_READ_THRESHOLD
                        (Optional) If the number of reads overlapping within a candidate transcript is below this threshold, throw out the transcript. Default: 2
  --assigned_read_ratio_threshold ASSIGNED_READ_RATIO_THRESHOLD
                        (Optional) The ratio of reads assigned solely to a candidate transcript to all reads overlapping within the same transcript. Keep transcripts that exceed this
                        ratio. Default: 0.2
  -o, --output_prefix OUTPUT_PREFIX
                        (Optional) Prefix of the output files to save results to. The script will append .candidates.bed, .bed and .gff3 to the filenames. Default: all_transcripts
  --candidates_only     (Optional) If enabled, skip the candidate transcript-filtering step and print out the candidate transcripts only
  -v, --verbose         (Optional) If enabled, print detailed step progress.
  --version             show program's version number and exit

--reads_bed file uses the first 6 standard BED columns (chr, start, end, name, score, strand)
--ranges_bed uses the first 4 standard BED columns (chr, start, end, strand)

The first output file ('.candidates.bed') is a 6-column BED file of all candidate transcripts for each chromsome. The 6 columns are (chr, start, end, name, score, strand).The second output file is a 6-column BED file of all predicted transcripts for each chromsome that passed our linear model fitting and prediction cutoffs. The columns are the same as the first file.The third output file is a 9-column GFF3 file of the same contents from the second BED file. The 'score' column is the normalized score from predicting the best transcripts for a region.
```

### Testing

A small test dataset (reads and ranges from the *E. coli* E2348/69 plasmids) is in `test-data/`, along with expected outputs. See `test-data/README.md` for how it was generated.

To run the regression tests, install `pytest` and run `python3 -m pytest tests` from the repository root.

### Citation

If you use this tool, please cite:

Mattick JSA, Bromley RE, Watson KJ, Adkins RS, Holt CI, Lebov JF, Sparklin BC, Tyson TS, Rasko DA, Dunning Hotopp JC. Deciphering transcript architectural complexity in bacteria and archaea. mBio. 2024. https://doi.org/10.1128/mbio.02359-24

Commands used for the manuscript are available at https://github.com/jdhotopp/tp.py-Direct-RNA-Sequencing-Manuscript-
