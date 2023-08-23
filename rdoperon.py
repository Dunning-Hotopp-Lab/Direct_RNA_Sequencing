#!/usr/bin/env python3

"""rdoperon.py (working name)

Predict operons based on a finding candidate start/stop sites,
and filter alternative transcripts that could make use of various start/stop combinations

"""

import argparse, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PoissonRegressor
from itertools import product

# Prevent warnings about parameter validation from popping up when computing the score
# Available in v1.3
#import sklearn
#sklearn.set_config(skip_parameter_validation = False)

BED_OUTPUT_COLS = ["chr","start","end","name","score","strand"]
GFF_OUTPUT_COLS = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]

def main():
    parser = argparse.ArgumentParser(
        description="Predict operons based on a finding candidate start/stop sites," \
            " and filter alternative transcripts that could make use of various start/stop combinations."
        , epilog="--reads_bed file uses the first 6 standard BED columns (chr, start, end, name, score, strand)\n" \
            "--ranges_bed uses the first 4 standard BED columns (chr, start, end, strand)\n\n" \
            "The first output file ('.candidates.bed') is a 6-column BED file of all candidate transcripts for each chromsome. The 6 columns are (chr, start, end, name, score, strand)."
            "The second output file is a 6-column BED file of all predicted transcripts for each chromsome that passed our linear model fitting and prediction cutoffs. The columns are the same as the first file."
            "The third output file is a 9-column GFF3 file of the same contents from the second BED file. The 'score' column is the normalized score from predicting the best transcripts for a region."
        , formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-r', '--reads_bed', type=str, required=True, help='Path to a position-sorted BED file of reads.')
    parser.add_argument('-d', '--ranges_bed', type=str, required=True, help="Path to a BED file of distict non-overlapping interval ranges per strand")
    parser.add_argument('-c', '--chromosome', type=str, required=False, help='Specific check on a chromsome ID. If not provided, will loop over all chromosomes')
    parser.add_argument('--min_depth', type=int, default=1, required=False, help='Minimum required depth for a distinct region to find candidate transcripts.')
    parser.add_argument('--max_depth', type=int, required=False, help='Maximum required depth for a distinct region to find candidate transcripts. If not set, max threshold is disabled.')
    parser.add_argument('--min_region_depth', type=int, default=2, required=False, help='Minimum required positional depth within a region. Reads mapping to positions that are below this threshold are tossed and new distinct subregions are created from the remaining runs of positions.')
    parser.add_argument("--candidate_offset_len", type=int, default=25, required=False, help="Some of the operon transcript candidates may have very close start or end coordinates. When encountering a start or end coordinate, ensure the next candidate position is at minimum <offset_length> bases away.")
    parser.add_argument("--depth_delta_threshold", type=int, default=1, help="If the absolute change of depth between two consecutive positions exceeds this number, consider this a potential start or stop site.")
    parser.add_argument("--assigned_read_ratio_threshold", type=float, default=0.2, help="The ratio of reads assigned solely to a candidate transcript to all assigned reads to the same transcript. Keep transcripts that exceed this ratio.")
    parser.add_argument('-o', "--output_prefix", type=str, default="all_transcripts", required=False, help="Prefix of the output files to save results to. The script will append .bed and .gff3 to the filenames")
    parser.add_argument("--candidates_only", action="store_true", help="If enabled, skip the linear-model fitting step for the candidate transcripts and print out the candidate transcripts only")
    parser.add_argument("-v", "--verbose", action="store_true", help="If enabled, print detailed step progress.")
    args = parser.parse_args()

    if args.min_depth < 0:
        sys.exit("--min_depth cannot be less than 0. Exiting")
    if args.max_depth < 1:
        sys.exit("--max_depth cannot be less than 1. Exiting")
    if args.depth_delta_threshold < 0:
        sys.exit("--depth_delta_threshold cannot be less than 0. Exiting")
    if args.assigned_read_ratio_threshold < 0 or args.assigned_read_ratio_threshold > 1:
        sys.exit("--assigned_read_ratio_threshold most be between 0 and 1. Exiting")

    distinct_ranges = args.ranges_bed
    dr_cols = ["chr","start","end", "strand"]
    distinct_ranges_df = pd.read_csv(distinct_ranges, sep="\t", header=None, names=dr_cols, usecols=[0,1,2,3])
    # Sorting is not necessary for this output
    distinct_ranges_df = distinct_ranges_df.sort_values(by=["chr", "strand", "start"]).reset_index(drop=True)
    distinct_ranges_df["region"] = distinct_ranges_df.index

    reads_file = args.reads_bed
    reads_cols = ["chr","start","end","name","score","strand"]
    reads_df = pd.read_csv(reads_file,  sep="\t", header=None, names=reads_cols, usecols=[0,1,2,3,4,5])
    reads_df = reads_df.sort_values(by=["chr", "strand", "start"]).reset_index(drop=True)

    transcript_df = pd.DataFrame(columns=BED_OUTPUT_COLS)
    candidate_df = pd.DataFrame(columns=BED_OUTPUT_COLS)
    annotation_df = pd.DataFrame(columns=GFF_OUTPUT_COLS)

    # Process each usable region
    read_ranges_grouped = distinct_ranges_df.groupby(["chr", "region"], sort=False)
    for name, subread_ranges_df in read_ranges_grouped:
        subread_ranges_s = subread_ranges_df.squeeze()

        subreads_df = assign_reads_to_region(subread_ranges_s, reads_df).copy()

        # Quick isolation of a single chromosome or region interval
        if args.chromosome and not name[0] == args.chromosome:
            continue

        # Filter only the reads that belong to this region.
        if len(subread_ranges_df) > 1:
            raise f"Found multiple entries for chr {name[0]} and region {name[1]}. Exiting."

        if len(subreads_df) < args.min_depth:
            print(f"Skipping region {name} because depth is under the min threshold of {args.min_depth}.")
            continue

        # Get ranges with valid depth (filter out ribosomes with high depth)
        if args.max_depth and len(subreads_df) > args.max_depth:
            print(f"Skipping region {name} because depth is over the max threshold of {args.max_depth}")
            continue

        subreads_df["region"] = name[1]
        if args.verbose:
            print(f"REGION {name}")

        region_transcript_df, region_annotation_df, region_candidate_df = predict_region(subreads_df, args.min_region_depth, args.candidate_offset_len, args.depth_delta_threshold, args.assigned_read_ratio_threshold, args.candidates_only, args.verbose)
        transcript_df = pd.concat([transcript_df, region_transcript_df], ignore_index=True)
        annotation_df = pd.concat([annotation_df, region_annotation_df], ignore_index=True)
        candidate_df = pd.concat([candidate_df, region_candidate_df], ignore_index=True)

    output_prefix = args.output_prefix

    candidate_df = candidate_df.sort_values(by=["chr", "strand", "start"])
    candidate_df.to_csv(f"{output_prefix}.candidates.bed", sep="\t", index=False, header=False)

    if not args.candidates_only:
        transcript_df = transcript_df.sort_values(by=["chr", "strand", "start"])
        transcript_df.to_csv(f"{output_prefix}.bed", sep="\t", index=False, header=False)

        annotation_df = annotation_df.sort_values(by=["seqid", "strand", "start"])
        annotation_df.to_csv(f"{output_prefix}.gff3", sep="\t", index=False, header=False)
        prepend_gff_version(f"{output_prefix}.gff3")

    exit()

def prepend_gff_version(filename):
    """Prepend GFF3 version tag."""
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("##gff-version 3\n")

def predict_region(reads_df:pd.DataFrame, min_region_depth:int, candidate_offset_len:int, depth_delta_threshold:int, assigned_read_ratio_threshold:float, only_candidates:bool, verbose:bool):
    # Each region belongs to a single chromosome and strand
    chromosome = reads_df["chr"].unique()[0]
    orientation = reads_df["strand"].unique()[0]

    # These should also have just one unique val too
    region_index = reads_df["region"].unique()[0]

    # NOTE: BED-files are non-inclusive at the stop coordinate
    # Also all coordinates will be left-most as start and right-most as end
    # For reverse strand, "end" is still non-inclusive (https://www.biostars.org/p/63263/#63265)

    range_start = reads_df["start"].min()
    range_end = reads_df["end"].max()
    total_range_s = pd.Series(list(range(range_start, range_end)))

    subtranscript_df = pd.DataFrame(columns=BED_OUTPUT_COLS)    # final predicted transcripts
    annotation_df = pd.DataFrame(columns=GFF_OUTPUT_COLS)
    candidate_df = pd.DataFrame(columns=BED_OUTPUT_COLS)    # all candidate transcripts

    # If only one read in the region interval, that will be the de facto operon
    if len(reads_df) < 2:
        operon_name = f"O_{region_index}_T1"
        reads_s = reads_df[["chr", "start", "end"]].squeeze()
        reads_s["start"] = range_start
        reads_s["end"] = range_end
        reads_s["name"] = operon_name
        reads_s["score"] = "."
        reads_s["strand"] = orientation
        subtranscript_df.loc[len(subtranscript_df)] = reads_s
        candidate_df.loc[len(candidate_df)] = reads_s

        gff_s = pd.Series(index=GFF_OUTPUT_COLS)
        gff_s["seqid"] = reads_s["chr"]
        gff_s["source"] = "rdoperon.py"
        gff_s["type"] = "transcript"
        gff_s["start"] = range_start + 1  # GFF3 are 1-indexed
        gff_s["end"] = range_end   # inclusive
        gff_s["score"] = 0
        gff_s["strand"] = orientation
        gff_s["phase"] = "."
        gff_s["attributes"] = f"ID={operon_name}"
        annotation_df.loc[len(annotation_df)] = gff_s

        return subtranscript_df, annotation_df, candidate_df

    # Depth within distinct region
    if verbose:
        print("\t- Region depth calculations")
    depth_df = create_depth_df(reads_df[["chr", "start", "end"]], total_range_s)

    # Filter any low-quality parts of the region
    # For now, I am including the extreme ends as it is easier to code
    # Also we only will do this once.
    good_region_depth_df = depth_df[depth_df["raw_depth"] >= min_region_depth].reset_index()
    bad_region_depth_df = depth_df[~(depth_df["raw_depth"] >= min_region_depth)].reset_index()
    reads_to_rm_df = pd.DataFrame(columns=reads_df.columns)
    # Find all reads to remove
    for pos in bad_region_depth_df["rel_position"]:
        reads_to_rm_df = pd.concat([reads_to_rm_df, reads_df[(reads_df["start"] <= pos) & (pos < reads_df["end"])]])
    # Remove from master regional read list
    reads_df = reads_df[~reads_df["name"].isin(reads_to_rm_df.drop_duplicates()["name"])]
    region_reads_df = reads_df.copy()

    # Split good region positions into new regions per consecutive sequence of numbers
    # Source -> https://stackoverflow.com/a/7353335
    good_pos = good_region_depth_df["rel_position"].to_numpy()
    # Create number of intervals (+1) where gap between current and next number is not 1
    new_regions = np.split(good_pos, np.where(np.diff(good_pos) != 1)[0]+1)
    split_counter = 0

    orig_region_index = region_index
    for region_arr in new_regions:
        # Update region start and end based on the good positions
        local_start = np.min(region_arr)
        local_end = np.max(region_arr)
        subregion_range_s = pd.Series(list(range(local_start, local_end+1)))    # local_end has valid depth, but
        # If subregion is just a single base, skip
        if not len(subregion_range_s):
            continue

        if len(new_regions) > 1:
            if verbose:
                print(f"\t- Region split {split_counter}")
                print(f"\t-- Range {local_start} - {local_end+1}")
            region_index = f"{orig_region_index}/{split_counter}"
            split_counter += 1

        # Reassign reads to the new sub-regions
        # There may be a situation where reads were removed that fit a subregion previously
        # but now no remaining reads align to that subregion.
        subread_ranges_s = pd.Series({
            "chr":chromosome
            , "start":local_start
            , "end": local_end + 1  # Subread range needs to be non-inclusive on end
            , "strand": orientation
            })
        reads_df = assign_reads_to_region(subread_ranges_s, region_reads_df)
        if reads_df.empty:
            if verbose:
                print(f"\t-- Removed subregion {region_index} as no reads were assigned to it.")
            continue

        # NOTE: Initially I thought if there was one region we should just keep original depth and ranges
        # but we would also need to restore the original set of reads too. I figured we would just leave be for now
        depth_df = create_depth_df(reads_df[["chr", "start", "end"]], subregion_range_s)

        #print("DEPTH_DF")
        #print(depth_df)

        # Get 1st derivative information based on changes in depth
        # 1st derivative is raw_delta
        # Some depths will be higher than others, so this is normalized in "derivative"
        if verbose:
            print("\t-- Change of depth calculations")
        slope_df = create_slope_df(depth_df)

        # Filter by whatever excceds absolute derivative threshold cutoff
        filtered_slope_df = filter_slope_df_by_threshold(slope_df, depth_delta_threshold)

        # Build candidate start and stop regions. The entire region range can be considered a transcript
        start_sites = get_candidate_starts(local_start, filtered_slope_df, candidate_offset_len)
        end_sites = get_candidate_ends(local_start, filtered_slope_df, candidate_offset_len)

        # Determine candidate transcripts using every combination of start/stop bins
        cartesian_pos_sites = list(product(start_sites, end_sites))
        transcript_candidate_data = [{"id": index, "start": pos[0], "end":pos[1]} for index, pos in enumerate(cartesian_pos_sites)]

        possible_transcripts_df = create_possible_transcripts_df(transcript_candidate_data)

        #print("POSSIBLE TRANSCRIPTS DF")
        #print(possible_transcripts_df)

        for row in possible_transcripts_df.itertuples():
            operon_name = f"O_{region_index}_T{row.Index}"
            start = row.start
            end = row.end
            score = "."
            transcript_s = pd.Series([chromosome, start, end, operon_name, score, orientation], index=["chr", "start", "end", "name",  "score", "strand"])
            transcript_s["name"] = operon_name
            candidate_df.loc[len(candidate_df)] = transcript_s

        if only_candidates:
            return subtranscript_df, annotation_df, candidate_df

        if verbose:
            print("\t-- Assigning reads to candidate transcripts")

        if len(possible_transcripts_df) == 1 \
            and possible_transcripts_df.iloc[0]["start"] == local_start \
            and possible_transcripts_df.iloc[0]["end"] == local_end:
                # if there is only one transcript and it spans the region, just use the region depth
                # since all reads were assigned to this region already
                final_transcripts_dict = {possible_transcripts_df.iloc[0]["id"] : 1}
        else:
            # Generate each individual transcript depth and model them with domains
            final_transcripts_dict = get_indiv_transcript_depths(reads_df, possible_transcripts_df, subregion_range_s, assigned_read_ratio_threshold)

        if verbose:
            print("\t-- Create linear models")

        #print("LINEAR MODELS DF")
        #print(linear_models_df)

        if verbose:
            print("\t-- Predicting best candidate transcripts")

        if verbose:
            print("\t-- Finalizing transcript list")

        final_transcripts_df = possible_transcripts_df[possible_transcripts_df["id"].isin((list(final_transcripts_dict.keys())))].copy()

        for row in final_transcripts_df.itertuples():
            operon_name = f"O_{region_index}_T{row.Index}"
            start = row.start
            end = row.end
            score = "."
            transcript_s = pd.Series([chromosome, start, end, operon_name, score, orientation], index=["chr", "start", "end", "name",  "score", "strand"])
            transcript_s["name"] = operon_name
            subtranscript_df.loc[len(subtranscript_df)] = transcript_s

            gff_s = pd.Series(index=GFF_OUTPUT_COLS)
            gff_s["seqid"] = chromosome
            gff_s["source"] = "rdoperon.py"
            gff_s["type"] = "transcript"
            gff_s["start"] = start + 1  # GFF3 are 1-indexed
            gff_s["end"] = end  # inclusive
            gff_s["score"] = round(final_transcripts_dict[row.Index], 2)
            gff_s["strand"] = orientation
            gff_s["phase"] = "."
            gff_s["attributes"] = f"ID={operon_name}"
            annotation_df.loc[len(annotation_df)] = gff_s
    return subtranscript_df, annotation_df, candidate_df

def get_indiv_transcript_depths(reads_df, possible_transcripts_df, total_range_s, assigned_read_ratio_threshold):
    """Get positional depth for each candidate transcript after assigning reads to a single candidate."""
    reads_shinking_df = reads_df.copy()
    final_transcripts = {}

    # Assign reads to transcript candidates to calculate depth and train model on it
    # ? Can we use a reducing command to shrink the reads_df
    for row in possible_transcripts_df.itertuples():
        # No reads left to assign
        if reads_shinking_df.empty:
            break
        # Interior reads fall within the real start and end of a given transcript candidate
        interior_reads_mask = (row.start <= reads_shinking_df["start"]) & (reads_shinking_df["end"] <= row.end)
        interior_reads_all_mask = (row.start <= reads_df["start"]) & (reads_df["end"] <= row.end)
        interior_reads_df = reads_shinking_df[interior_reads_mask]
        interior_reads_all_df = reads_df[interior_reads_all_mask]
        # Shrink the list of remaining reads
        reads_shinking_df = reads_shinking_df[~(reads_shinking_df["name"].isin(interior_reads_df["name"]))]

        # No interior reads assigned to this transcript
        if interior_reads_df.empty:
            continue

        ratio = len(interior_reads_df) / len(interior_reads_all_df)

        if ratio > assigned_read_ratio_threshold:
            final_transcripts[row.id] = ratio
    return final_transcripts

def get_candidate_ends(range_end, filtered_slope_df, candidate_offset_len):
    end_sites = {range_end}
    end_sites.update(filtered_slope_df.loc[filtered_slope_df["direction"] == -1, "rel_position"].unique())
    sorted_ends = np.array(sorted(end_sites, reverse=True))
    final_ends = set()

    max_end = max(sorted_ends)
    final_ends.add(max_end)

    # Only take ends that are not close together
    for pos in sorted_ends:
        if pos + candidate_offset_len <= max_end:
            max_end = pos
            final_ends.add(max_end)

    return final_ends

def get_candidate_starts(range_start, filtered_slope_df, candidate_offset_len):
    start_sites = {range_start}
    start_sites.update(filtered_slope_df.loc[filtered_slope_df["direction"] == 1, "rel_position"].unique())

    sorted_starts = np.array(sorted(start_sites))
    final_starts = set()

    min_start = min(sorted_starts)
    final_starts.add(min_start)

    # Only take starts that are not close together
    for pos in sorted_starts:
        if min_start + candidate_offset_len <= pos:
            min_start = pos
            final_starts.add(min_start)

    return final_starts

def filter_slope_df_by_threshold(slope_df, threshold):
    # Breaking the dataframe into positive and negative changes in depth
    pos_direction_mask = slope_df["raw_delta"] > threshold
    neg_direction_mask = slope_df["raw_delta"] < (-1 * threshold)
    slope_df["direction"] = 0   # ? probably not necessary. Sanity-check addition
    slope_df.loc[pos_direction_mask, "direction"] = 1
    slope_df.loc[neg_direction_mask, "direction"] = -1
    filtered_slope_df = slope_df.loc[~(slope_df["direction"] == 0), ["rel_position", "direction"]]
    return filtered_slope_df

def assign_reads_to_region(subread_ranges_s, reads_df):
    # Assign this read to a region
    # Read must fall within region boundaries and be in the same strand
    mask = (subread_ranges_s["start"] <= reads_df["start"]) & (reads_df["end"] <= subread_ranges_s["end"]) & (subread_ranges_s["strand"] == reads_df["strand"])
    return reads_df[mask]

def create_possible_transcripts_df(transcript_dict):
    possible_transcripts_df = pd.DataFrame(transcript_dict)
    possible_transcripts_df["length"] = possible_transcripts_df["end"] - possible_transcripts_df["start"]

    # This is important. Need to sort be smallest to largest, so that reads can be assigned to transcripts appropriately
    possible_transcripts_df = possible_transcripts_df[possible_transcripts_df["length"] > 0].sort_values(by="length")
    return possible_transcripts_df

def create_slope_df(depth_df):
    slope_df = depth_df[["transcript", "rel_position", "raw_depth"]].sort_values(by="rel_position")
    # Adding a 0-depth entry just beyond left-most area (https://stackoverflow.com/a/45466227)
    slope_df.loc[-1] = [slope_df.loc[0, "transcript"], np.min(slope_df["rel_position"])-1, 0]
    slope_df.index = slope_df.index + 1
    slope_df = slope_df.sort_index()
    # Get the next position for the slope calculation (final row will be filled with 0)
    slope_df["next_raw_depth"] = slope_df["raw_depth"].shift(periods=-1).fillna(0)
    # change between positions
    slope_df["raw_delta"] = slope_df["next_raw_depth"] - slope_df["raw_depth"]
    # average depth between positions
    #slope_df["split_depth"] = (slope_df["next_raw_depth"] + slope_df["raw_depth"]) / 2
    # Normalizing change depending on the quantity of depth
    #slope_df["derivative"] = slope_df["raw_delta"] / slope_df["split_depth"]
    slope_df = slope_df.dropna()

    # Exclude positions with no depth (incl. out-of-region ones introduced at top of function)
    depth_filter_mask = slope_df["raw_depth"] > 0
    return slope_df[depth_filter_mask]

def create_depth_df(reads_df:pd.DataFrame, total_range_s:pd.Series):
    depth_df = pd.DataFrame(total_range_s, columns=["rel_position"])
    depth_df["transcript"] = reads_df["chr"].unique()[0]    # Since we grouped by chromosome, all reads should be in same chromosome
    raw_depth_list = [len(reads_df[(reads_df["start"] <= pos) & (pos < reads_df["end"])]) for pos in depth_df["rel_position"].tolist()]
    depth_df["raw_depth"] = pd.Series(raw_depth_list)
    return depth_df

if __name__ == '__main__':
    main()
