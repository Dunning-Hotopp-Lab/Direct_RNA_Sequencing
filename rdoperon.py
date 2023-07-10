#!/usr/bin/env python3

"""rdoperon.py (working name)

Predict operons based on a finding candidate start/stop sites,
and filter alternative transcripts that could make use of various start/stop combinations

"""

import argparse, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import product

BED_OUTPUT_COLS = ["chr","start","end","name","score","strand"]
GFF_OUTPUT_COLS = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]

model = LinearRegression()

def main():
    parser = argparse.ArgumentParser(
        description="Predict operons based on a finding candidate start/stop sites," \
            " and filter alternative transcripts that could make use of various start/stop combinations."
        , epilog="--reads_bed file uses the first 6 standard BED columns (chr, start, end, name, score, strand)\n" \
            "--ranges_bed uses the first 4 standard BED columns (chr, start, end, strand)\n\n" \
            "The first output file is a 6-column BED file of all candidate alternative transcripts for each chromsome. The 6 columns are (chr, start, end, name, score, strand)"
            "The second output file is a 9-column GFF3 file of the same contents from the BED file. The 'score' column is the normalized score from predicting the best transcripts for a region."
        , formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-r', '--reads_bed', type=str, required=True, help='Path to a position-sorted BED file of reads.')
    parser.add_argument('-d', '--ranges_bed', type=str, required=True, help="Path to a BED file of distict non-overlapping interval ranges per strand")
    parser.add_argument('-c', '--chromosome', type=str, required=False, help='Specific check on a chromsome ID. If not provided, will loop over all chromosomes')
    parser.add_argument('--min_depth', type=int, default=1, required=False, help='Minimum required depth for a distinct region to find candidate transcripts.')
    parser.add_argument('--max_depth', type=int, required=False, help='Maximum required depth for a distinct region to find candidate transcripts. If not set, max threshold is disabled.')
    parser.add_argument('--min_region_depth', type=int, default=2, required=False, help='Minimum required positional depth within a region. Reads mapping to positions that are below this threshold are tossed and new distinct subregions are created from the remaining runs of positions.')
    parser.add_argument("--bin_size", type=int, default=25, required=False, help="Some of the operon transcript candidates may have very close start or end coordinates. Perform binning of the specified size on the entire transcript region and take the min start or max end positions per bin amongst the candidate positions.")
    parser.add_argument("--num_stdev", type=float, default=2.0, required=False, help="Set the number of standard deviations to filter potential start and stop sites.")
    parser.add_argument('-o', "--output_prefix", type=str, default="all_transcripts", required=False, help="Prefix of the output files to save results to. The script will append .bed and .gff3 to the filenames")
    parser.add_argument("-v", "--verbose", action="store_true", help="If enabled, print detailed step progress.")
    args = parser.parse_args()

    if args.min_depth < 0:
        sys.exit("--min_depth cannot be less than 0. Exiting")
    if args.max_depth < 1:
        sys.exit("--max_depth cannot be less than 1. Exiting")
    if args.num_stdev < 0:
        sys.exit("--num_stdev cannot be less than 0. Exiting")

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
    annotation_df = pd.DataFrame(columns=GFF_OUTPUT_COLS)

    # Process each usable region
    read_ranges_grouped = distinct_ranges_df.groupby(["chr", "region"], sort=False)
    for name, subread_ranges_df in read_ranges_grouped:
        # Quick isolation of a single chromosome or region interval
        if args.chromosome and not name[0] == args.chromosome:
            continue

        # Filter only the reads that belong to this region.
        if len(subread_ranges_df) > 1:
            raise f"Found multiple entries for chr {name[0]} and region {name[1]}. Exiting."

        subread_ranges_s = subread_ranges_df.squeeze()
        subreads_df = assign_reads_to_region(subread_ranges_s, reads_df).copy()

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

        region_transcript_df, region_annotation_df = predict_region(subreads_df, args.min_region_depth, args.bin_size, args.num_stdev, args.verbose)
        transcript_df = pd.concat([transcript_df, region_transcript_df], ignore_index=True)
        annotation_df = pd.concat([annotation_df, region_annotation_df], ignore_index=True)

    output_prefix = args.output_prefix
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

def predict_region(reads_df:pd.DataFrame, min_region_depth:int, bin_size:int, num_stdev:float, verbose:bool):
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

    subtranscript_df = pd.DataFrame(columns=BED_OUTPUT_COLS)
    annotation_df = pd.DataFrame(columns=GFF_OUTPUT_COLS)

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

        return subtranscript_df, annotation_df

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
                print(f"\t- Removed subregion {region_index} as no reads were assigned to it.")
            continue

        # NOTE: Initially I thought if there was one region we should just keep original depth and ranges
        # but we would also need to restore the original set of reads too. I figured we would just leave be for now
        depth_df = create_depth_df(reads_df[["chr", "start", "end"]], subregion_range_s)

        #print("DEPTH_DF")
        #print(depth_df)

        # Get 1st and 2nd derivative information based on changes in depth
        # 1st derivative is norm_delta
        # Some depths will be higher than others, so this is normalized in "norm_derivative"
        if verbose:
            print("\t- Change of depth calculations")
        slope_df = create_slope_df(depth_df)
        raw_depth_s = slope_df["raw_depth"]

        # Filter by whatever excceds + and - standard deviation cutoff
        filtered_slope_df = filter_slope_df_by_std(slope_df, num_stdev)

        # Build candidate start and stop regions. The entire region range can be considered a transcript
        start_sites = get_candidate_starts(local_start, filtered_slope_df, bin_size)
        end_sites = get_candidate_ends(local_start, filtered_slope_df, bin_size)

        # Determine candidate transcripts using every combination of start/stop bins
        cartesian_pos_sites = list(product(start_sites, end_sites))
        transcript_candidate_data = [{"id": index, "start": pos[0], "end":pos[1]} for index, pos in enumerate(cartesian_pos_sites)]

        possible_transcripts_df = create_possible_transcripts_df(transcript_candidate_data)

        #print("POSSIBLE TRANSCRIPTS DF")
        #print(possible_transcripts_df)

        if verbose:
            print("\t- Assigning reads to candidate transcripts")

        if len(possible_transcripts_df) == 1 \
            and possible_transcripts_df.iloc[0]["start"] == local_start \
            and possible_transcripts_df.iloc[0]["end"] == local_end:
                # if there is only one transcript and it spans the region, just use the region depth
                # since all reads were assigned to this region already
                transcript_id = "T_{}".format(possible_transcripts_df.iloc[0]["id"])
                depth_subtranscripts_df = pd.DataFrame(subregion_range_s, columns=["rel_position"])
                depth_subtranscripts_df[transcript_id] = depth_df["raw_depth"]
        else:
            # Generate each individual transcript depth and model them with domains
            depth_subtranscripts_df = get_indiv_transcript_depths(reads_df, possible_transcripts_df, subregion_range_s)

        if verbose:
            print("\t- Create linear models")

        linear_models_df = create_linear_models(possible_transcripts_df, depth_subtranscripts_df)

        #print("LINEAR MODELS DF")
        #print(linear_models_df)

        if verbose:
            print("\t- Predicting best candidate transcripts")

        # Apply the linear model iteratively until a convergence happens, taking the best transcript at each step
        # Output key - transcript, value - normalized error score
        final_transcripts_dict = find_best_transcripts(linear_models_df, raw_depth_s, subregion_range_s)

        if verbose:
            print("\t- Finalizing transcript list")

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
    return subtranscript_df, annotation_df

def get_indiv_transcript_depths(reads_df, possible_transcripts_df, total_range_s):
    reads_shinking_df = reads_df.copy()
    depth_subtranscripts_df = pd.DataFrame(total_range_s, columns=["rel_position"])

    # This is to prevent a PerformanceWarning in pandas where columns are repeatedly inserted into
    # Source -> https://stackoverflow.com/a/75776863
    depth_subtranscripts_dict = {}

    # Assign reads to transcript candidates to calculate depth and train model on it
    # ? Can we use a reducing command to shrink the reads_df
    for row in possible_transcripts_df.itertuples():
        # No reads left to assign
        if reads_shinking_df.empty:
            break
        transcript_id = "T_{}".format(str(row.id))
        # Interior reads fall within the real start and end of a given transcript candidate
        interior_reads_mask = (row.start <= reads_shinking_df["start"]) & (reads_shinking_df["end"] < row.end)
        interior_reads_df = reads_shinking_df[interior_reads_mask]
        # Shrink the list of remaining reads
        reads_shinking_df = reads_shinking_df[~(reads_shinking_df["name"].isin(interior_reads_df["name"]))]

        # No interior reads assigned to this transcript
        if interior_reads_df.empty:
            continue

        # Calculate base depth just among interior reads for a transcript

        # SLOW VERSION BUT USES LESS MEMORY
        #raw_depth_list = [len(interior_reads_df[(interior_reads_df["start"] <= pos) & (pos < interior_reads_df["end"])]) for pos in depth_subtranscripts_df["rel_position"].to_list()]
        #depth_subtranscripts_dict[transcript_id] = pd.Series(raw_depth_list)

        # FAST VERSION BUT CAN BE MEMORY-INTENSIVE
        depth_reads_df = pd.merge(depth_subtranscripts_df[["rel_position"]], interior_reads_df[["start", "end"]], how="cross")
        depth_subtranscripts_dict[transcript_id] = pd.Series(get_raw_depth(depth_reads_df))


    temp_df = pd.DataFrame(data=depth_subtranscripts_dict.values(), index=depth_subtranscripts_dict.keys()).transpose()
    depth_subtranscripts_df = pd.concat([depth_subtranscripts_df, temp_df], axis="columns")
    return depth_subtranscripts_df

def find_best_transcripts(linear_models_df, raw_depth_s, total_range_s):
    """Add each linear model stepwise until additions of the linear models do not improve the score."""
    # Initial condition of linear model
    linear_models_leftover_df = linear_models_df.copy()
    depth_frame_df = pd.DataFrame(total_range_s, columns=["pos"])
    depth_frame_df["depth"] = 0
    depth_frame_df["actual_depth"] = raw_depth_s

    final_transcripts = dict()

    # If no more transcript candidates are left to process, break
    while not linear_models_leftover_df.empty:
        baseline_error = get_sum_of_squares(depth_frame_df["actual_depth"], depth_frame_df["depth"])
        data = [{"id":row.id, "a":row.a, "b":row.b, "dstart":row.dstart, "dend":row.dend} for row in linear_models_leftover_df.itertuples()]
        indiv_assessment_frame_df = pd.DataFrame(data, index=linear_models_leftover_df.index)

        # Perform predictions for each transcript candidate
        for row in indiv_assessment_frame_df.itertuples():
            # ? could we use model.predict to get this instead?
            predicted_frame_df = predict_linear_domain(row, total_range_s)
            normalize_linear_domain_depth(depth_frame_df, predicted_frame_df)
            indiv_assessment_frame_df.loc[row.Index, "sum_error"] = get_sum_of_squares(depth_frame_df["actual_depth"], predicted_frame_df["predicted_depth"])
            indiv_assessment_frame_df.loc[row.Index, "depth_fraction"]  = len(predicted_frame_df[predicted_frame_df["depth"] > 0]) / len(predicted_frame_df)

        # I think the idea here is that if "sum_error" -> "u", "baseline_error" -> "v" and coefficient of determination (R^2) -> 1 - u/v
        # then a bad model would have a negative R^2 because the "sum_error" was worse than "baseline"

        # Normalize, then keep all transcript models that are better than the baseline in terms of their prediction
        indiv_assessment_frame_df["norm_error"] = (baseline_error - indiv_assessment_frame_df["sum_error"]) * indiv_assessment_frame_df["depth_fraction"]
        indiv_assessment_frame_df = indiv_assessment_frame_df[indiv_assessment_frame_df["norm_error"] > 0]

        # If no more transcript models pass error cutoff, exit
        if indiv_assessment_frame_df.empty:
            break

        # Predict again on the best transcript model.
        # Use this predicted depth as new baseline error for the next iteration
        indiv_assessment_frame_df = indiv_assessment_frame_df.sort_values(by="norm_error", ascending=False)
        best_transcript = indiv_assessment_frame_df.iloc[0]["id"]
        final_transcripts[best_transcript] = indiv_assessment_frame_df.iloc[0]["norm_error"]

        best_leftover_transcript_s = linear_models_leftover_df[linear_models_leftover_df["id"] == best_transcript].squeeze()

        predicted_frame_df = predict_linear_domain(best_leftover_transcript_s, total_range_s)
        normalize_linear_domain_depth(depth_frame_df, predicted_frame_df)
        depth_frame_df["depth"] = predicted_frame_df["predicted_depth"]

        linear_models_leftover_df = linear_models_leftover_df[~(linear_models_leftover_df["id"] == best_transcript)]

    return final_transcripts

def get_candidate_ends(range_end, filtered_slope_df, bin_size):
    end_sites = {range_end}
    end_sites.update(filtered_slope_df.loc[filtered_slope_df["direction"] == -1, "rel_position"].unique())
    # Bin start and end sites to eliminate closely-grouped positions
    sorted_ends = np.array(sorted(end_sites))
    final_ends = set()

    for pos in range(min(sorted_ends), max(sorted_ends)+1, bin_size):
        bin_ends = sorted_ends[(pos <= sorted_ends) & (sorted_ends <= pos+bin_size)]
        if len(bin_ends):
            final_ends.add(np.max(bin_ends))
    return final_ends

def get_candidate_starts(range_start, filtered_slope_df, bin_size):
    start_sites = {range_start}
    start_sites.update(filtered_slope_df.loc[filtered_slope_df["direction"] == 1, "rel_position"].unique())
    # Bin start and end sites to eliminate closely-grouped positions
    sorted_starts = np.array(sorted(start_sites))
    final_starts = set()

    for pos in range(min(sorted_starts), max(sorted_starts)+1, bin_size):
        bin_starts = sorted_starts[(pos <= sorted_starts) & (sorted_starts <= pos+bin_size)]
        if len(bin_starts):
            final_starts.add(np.min(bin_starts))
    return final_starts

def filter_slope_df_by_std(slope_df, num_stdev):
    # Breaking the dataframe into positive and negative normalized derivatives
    # NOTE: We are not filtering st dev away from the pos and neg means, but rather from 0
    std_pos, std_neg = get_std(slope_df)
    std_pos_filter = std_pos * num_stdev
    std_pos_filter_mask = slope_df["norm_derivative"] >= std_pos_filter
    std_neg_filter = std_neg * -num_stdev
    std_neg_filter_mask = slope_df["norm_derivative"] <= std_neg_filter

    filtered_slope_df = slope_df.loc[(std_pos_filter_mask) | (std_neg_filter_mask), ["rel_position", "direction"]]
    return filtered_slope_df

def get_std(slope_df):
    pos_direction_mask = slope_df["norm_derivative"] > 0
    neg_direction_mask = slope_df["norm_derivative"] < 0
    slope_df["direction"] = 0
    slope_df.loc[pos_direction_mask, "direction"] = 1
    slope_df.loc[neg_direction_mask, "direction"] = -1
    std_pos = slope_df.loc[pos_direction_mask, "norm_derivative"].std(skipna=True)
    std_neg = slope_df.loc[neg_direction_mask, "norm_derivative"].std(skipna=True)
    return std_pos, std_neg

def create_linear_models( possible_transcripts_df, depth_subtranscripts_df):
    linear_models_df = pd.DataFrame(columns=["id", "a", "b", "dstart", "dend"])
    # Make linear models
    for row in possible_transcripts_df.itertuples():
        transcript_id = "T_{}".format(str(row.id))

        # Do not want transcripts with no reads assigned
        if transcript_id not in depth_subtranscripts_df.columns:
            continue

        # Filter regions where the bases are within the real start and end boundaries
        rel_position_mask = (row.start <= depth_subtranscripts_df["rel_position"]) & (depth_subtranscripts_df["rel_position"] < row.end)

        # Only train using transcripts with valid depth (log-transformed) and are within our region
        modeling_depth_df = depth_subtranscripts_df[rel_position_mask].copy()
        modeling_depth_df["log"] = 0
        modeling_depth_df = modeling_depth_df[modeling_depth_df[transcript_id] > 0]
        modeling_depth_df["log"] = np.log10(modeling_depth_df[transcript_id])

        if modeling_depth_df.empty:
            continue

        # Fit linear model (independent x val, dependent y val) and store results in-place
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
        # ! function "x" arg takes 2D array.
        model.fit(modeling_depth_df[["rel_position"]], modeling_depth_df["log"])

        # Linear mdel information for each candidate transcript
        # NOTE: "a" and "b" based on y=ax+b
        linear_models_df.loc[row.Index, "id"] = int(row.id)
        linear_models_df.loc[row.Index, "a"] = model.coef_[0]
        linear_models_df.loc[row.Index, "b"] = model.intercept_
        linear_models_df.loc[row.Index, "dstart"] = row.start
        linear_models_df.loc[row.Index, "dend"] = row.end
    return linear_models_df

def assign_reads_to_region(subread_ranges_s, reads_df):
    # Assign this read to a region
    # Read must fall within region boundaries and be in the same strand
    mask = (subread_ranges_s["start"] <= reads_df["start"]) & (reads_df["end"] <= subread_ranges_s["end"]) & (subread_ranges_s["strand"] == reads_df["strand"])
    return reads_df[mask]

def calc_norm_range_sum(df:pd.DataFrame):
    # This feels hacky. Need to return a series for the "apply" but switching to numpy loses the indexes
    return pd.Series(df["sum_norm_slope"].abs() / df["sum_norm_slope"].abs().max(), index=df.index)

def normalize_linear_domain_depth(depth_frame_df, predicted_frame_df):
    # Adjust depth to be in line with scale of recorded actual depth by position
    # NOTE: Edits DataFrame in-place
    predicted_frame_df["predicted_depth"] = 10**predicted_frame_df["depth"]
    predicted_frame_df.loc[predicted_frame_df["depth"] == 0, "predicted_depth"] = 0
    predicted_frame_df["predicted_depth"] += depth_frame_df["depth"]

def get_sum_of_squares(actual_depth:pd.Series, depth:pd.Series):
    """Return residual sum of squares"""
    return ((actual_depth - depth)**2).sum()

def create_possible_transcripts_df(transcript_dict):
    possible_transcripts_df = pd.DataFrame(transcript_dict)
    possible_transcripts_df["length"] = possible_transcripts_df["end"] - possible_transcripts_df["start"]

    # This is important. Need to sort be smallest to largest, so that reads can be assigned to transcripts appropriately
    possible_transcripts_df = possible_transcripts_df[possible_transcripts_df["length"] > 0].sort_values(by="length")
    return possible_transcripts_df

def create_slope_df(depth_df):
    slope_df = depth_df[["transcript", "rel_position", "norm_depth", "raw_depth"]].sort_values(by="rel_position")
    # Adding a 0-depth entry just beyond left-most area (https://stackoverflow.com/a/45466227)
    slope_df.loc[-1] = [slope_df.loc[0, "transcript"], np.min(slope_df["rel_position"])-1, 0, 0]
    slope_df.index = slope_df.index + 1
    slope_df = slope_df.sort_index()
    # Get the next position for the slope calculation (final row will be filled with 0)
    slope_df["next_raw_depth"] = slope_df["raw_depth"].shift(periods=-1).fillna(0)
    slope_df["next_norm_depth"] = slope_df["norm_depth"].shift(periods=-1).fillna(0)
    # change between positions
    slope_df["norm_delta"] = (slope_df["next_raw_depth"]/10 * slope_df["next_norm_depth"]) - (slope_df["raw_depth"]/10 * slope_df["norm_depth"])
    # average norm depth between positions
    slope_df["norm_split_depth"] = (slope_df["norm_depth"] + slope_df["next_norm_depth"]) / 2
    # Normalizing change depending on the quantity of depth
    slope_df["norm_derivative"] = slope_df["norm_delta"] / slope_df["norm_split_depth"]
    slope_df = slope_df.dropna()
    return slope_df

def create_depth_df(reads_df:pd.DataFrame, total_range_s:pd.Series):
    depth_df = pd.DataFrame(total_range_s, columns=["rel_position"])
    depth_df["transcript"] = reads_df["chr"].unique()[0]    # Since we grouped by chromosome, all reads should be in same chromosome
    raw_depth_list = [len(reads_df[(reads_df["start"] <= pos) & (pos < reads_df["end"])]) for pos in depth_df["rel_position"].tolist()]
    depth_df["raw_depth"] = pd.Series(raw_depth_list)

    # normalized depth
    depth_df["norm_depth"] = depth_df["raw_depth"] / max(depth_df["raw_depth"])
    return depth_df

def get_raw_depth(depth_reads_df):
    """Return depth for this particular base (number of reads with this base in its interval)."""
    # depth_reads_df is from a cross merge of depth_df and reads_df
    depth_reads_df["in_range"] = depth_reads_df["rel_position"].between(depth_reads_df["start"], depth_reads_df["end"], inclusive="left")
    return depth_reads_df.groupby("rel_position")["in_range"].sum().to_list()  # to list, otherwise indexes don't match

def predict_linear_domain(transcript_s, total_range_s):
    """Evaluate depth for each position in the region using this transcript's linear model"""
    # row - dstart/dend - transcript domain region
    # range_start/range_end - distinct region range
    # row - a - coefficient
    # row - b - y-intercept

    depth_frame_df = pd.DataFrame(total_range_s, columns=["rel_position"])
    depth_frame_df["depth"] = 0

    # NOTE: Explored using sckit-learn LinearRegression.predict, but it was slightly slower due to extra checks
    depth_frame_df["in_range"] = depth_frame_df["rel_position"].between(transcript_s.dstart, transcript_s.dend, inclusive="left")
    in_range_df = depth_frame_df[depth_frame_df["in_range"]]

    # prediction is y = ax + b
    depth_frame_df.loc[depth_frame_df["in_range"], "depth"] = (in_range_df["rel_position"] * transcript_s.a) + transcript_s.b
    return depth_frame_df


if __name__ == '__main__':
    main()
