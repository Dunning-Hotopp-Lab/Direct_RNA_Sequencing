#!/usr/bin/env python3

"""rdoperon.py (working name)

Predict operons based on a finding candidate start/stop sites,
and filter alternative transcripts that could make use of various start/stop combinations

"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from math import ceil
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
    parser.add_argument('--min_depth', type=int, default=0, required=False, help='Minimum required depth for a distinct region to find candidate transcripts.')
    parser.add_argument('--max_depth', type=int, required=False, help='Maximum required depth for a distinct region to find candidate transcripts. If not set, max threshold is disabled.')
    parser.add_argument('--min_normalized_slope', type=float, default=0.5, help='Minimum normalized depth threshold for deteremining start and stop bin candidiates.')
    parser.add_argument('-b', "--num_bins", type=int, default=100, required=False, help="Set number of total bins per region.")
    parser.add_argument('-o', "--output_prefix", type=str, default="all_transcripts", required=False, help="Prefix of the output files to save results to. The script will append .bed and .gff3 to the filenames")
    parser.add_argument("-v", "--verbose", action="store_true", help="If enabled, print detailed step progress.")
    args = parser.parse_args()

    distinct_ranges = args.ranges_bed
    dr_cols = ["chr","start","end", "strand"]
    distinct_ranges_df = pd.read_csv(distinct_ranges, sep="\t", header=None, names=dr_cols, usecols=[0,1,2,3])
    distinct_ranges_df = distinct_ranges_df.sort_values(by=["strand", "start"]).reset_index(drop=True)
    distinct_ranges_df["width"] = distinct_ranges_df["end"] - distinct_ranges_df["start"]   # BED end-coords are non-inclusive
    distinct_ranges_df["region"] = distinct_ranges_df.index

    reads_file = args.reads_bed
    reads_cols = ["chr","start","end","name","score","strand"]
    reads_df = pd.read_csv(reads_file,  sep="\t", header=None, names=reads_cols, usecols=[0,1,2,3,4,5])

    transcript_df = pd.DataFrame(columns=BED_OUTPUT_COLS)
    annotation_df = pd.DataFrame(columns=GFF_OUTPUT_COLS)

    # Process each usable region
    read_ranges_grouped = distinct_ranges_df.groupby(["chr", "region"])
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

        bin_width = subread_ranges_s["width"] / args.num_bins

        region_transcript_df, region_annotation_df = predict_region(subreads_df, bin_width, args.num_bins, args.min_normalized_slope, args.verbose)
        transcript_df = pd.concat([transcript_df, region_transcript_df], ignore_index=True)
        annotation_df = pd.concat([annotation_df, region_annotation_df], ignore_index=True)

    output_prefix = args.output_prefix
    transcript_df = transcript_df.sort_values(by="start")
    transcript_df.to_csv(f"{output_prefix}.bed", sep="\t", index=False, header=False)

    annotation_df = annotation_df.sort_values(by="start")
    annotation_df.to_csv(f"{output_prefix}.gff3", sep="\t", index=False, header=False)
    prepend_gff_version(f"{output_prefix}.gff3")
    exit()

def prepend_gff_version(filename):
    """Prepend GFF3 version tag."""
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("##gff-version 3\n")

def predict_region(reads_df:pd.DataFrame, bin_width:float, num_bins:int, min_normalized_slope:float, verbose:bool):
    # NOTE: From my understanding, each "region" interval is only on a single strand
    orientation = reads_df["strand"].unique()[0]

    # These should also have just one unique val too
    region_index = reads_df["region"].unique()[0]

    # NOTE: Squeeze turns a Dataframw with a single column or row into a Series
    # and turns a single-element Series or DataFrame into a scalar
    # Nice when you don't know the index labels
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.squeeze.html

    subtranscript_df = pd.DataFrame(columns=BED_OUTPUT_COLS)
    annotation_df = pd.DataFrame(columns=GFF_OUTPUT_COLS)

    # If only one read in the region interval, that will be the de facto operon
    if len(reads_df) < 2:
        operon_name = f"O_{region_index}_T1"
        reads_s = reads_df[["chr", "start", "end"]].squeeze()
        reads_s["start"] = int(reads_s["start"])
        reads_s["end"] = int(reads_s["end"])
        reads_s["name"] = operon_name
        reads_s["score"] = "."
        reads_s["strand"] = orientation
        subtranscript_df.loc[len(subtranscript_df)] = reads_s

        gff_s = pd.Series(index=GFF_OUTPUT_COLS)
        gff_s["seqid"] = reads_s["chr"]
        gff_s["source"] = "rdoperon.py"
        gff_s["type"] = "transcript"
        gff_s["start"] = reads_s["start"] + 1  # GFF3 are 1-indexed
        gff_s["end"] = reads_s["end"]   # inclusive
        gff_s["score"] = 0
        gff_s["strand"] = orientation
        gff_s["phase"] = "."
        gff_s["attributes"] = f"ID={operon_name}"
        annotation_df.loc[len(annotation_df)] = gff_s

        return subtranscript_df, annotation_df

    # Calculate bin categories
    range_start = reads_df["start"].min()
    range_end = (reads_df["end"].max() + ceil(bin_width))
    total_range_s = pd.Series(list(range(range_start, range_end)))
    # NOTE: The original R script actually ended up with one more bin when "findIntervals" was run.
    # I think the idea was to have one extra bin for the extra rollover to "tend"
    bin_labels = pd.cut(total_range_s, bins=num_bins+1, right=False, labels=False)

    # Depth
    if verbose:
        print("\t- Depth calculations")
    depth_df = create_depth_df(reads_df[["chr", "start", "end"]], total_range_s, bin_labels)
    valid_depth_mask = depth_df["raw_depth"] >= 1
    first_bin = depth_df[valid_depth_mask]["bin"].min()
    last_bin = depth_df[valid_depth_mask]["bin"].max()

    if first_bin == last_bin:
        last_bin += 1

    #print("DEPTH_DF")
    #print(depth_df)

    # Candidate start and stop regions
    start_site_bins = {first_bin if orientation == "+" else last_bin}
    stop_site_bins = {last_bin if orientation == "+" else first_bin}

    # Get 1st and 2nd derivative information based on differences in depth
    if verbose:
        print("\t- Depth of change calculations")
    slope_df = create_slope_df(depth_df)

    # Get cumulative amount of depth slope change for each bin
    bin_slope_df = slope_df.groupby("bin")["norm_slope"].sum().reset_index().rename(columns={"norm_slope":"sum_norm_slope"})

    #print("BIN_SLOPE_DF")
    #print(bin_slope_df)

    # Filter slopes that are greater than the threshold "factor"
    factor = max(bin_slope_df["sum_norm_slope"].abs().max()/20, min_normalized_slope)
    filter_slope_mask = bin_slope_df["sum_norm_slope"].abs() > factor
    filtered_bin_slope_df = bin_slope_df[filter_slope_mask].copy()  # avoids SettingWithCopyWarning

    # If any bin intervals pass filter, determine number of slope inversions
    if verbose:
        print("\t- Predicting start and stop candidate bins")
    if not filtered_bin_slope_df.empty:
        filtered_bin_slope_df["sign"] = 1 # number of slope changes

        is_positive = True if filtered_bin_slope_df.iloc[0]["sum_norm_slope"] >= 0 else False
        current_sign_count = 1
        # ? Can we optimize this
        for row in filtered_bin_slope_df.itertuples():
            if is_positive:
                if row.sum_norm_slope < 0:
                    is_positive = False
                    current_sign_count += 1
            else:
                if row.sum_norm_slope > 0:
                    is_positive = True
                    current_sign_count += 1
            filtered_bin_slope_df.loc[row.Index,"sign"] = current_sign_count

        # Prepate to take second derivative of depth changes
        if len(filtered_bin_slope_df["sign"].unique()) == 1:
            # "groupby" returns a Series instead of a DataFrame if only one group present.
            filtered_bin_slope_df["norm_range_sum"] = calc_norm_range_sum(filtered_bin_slope_df)
        else:
            filtered_bin_slope_df["norm_range_sum"] = filtered_bin_slope_df.groupby("sign", group_keys=False) \
                .apply(calc_norm_range_sum)

        #Now have a list of start and stop candidate bins by looking at their 1st and 2nd derivative values
        norm_range_mask = filtered_bin_slope_df["norm_range_sum"] > min_normalized_slope
        start_candidate_mask = (filtered_bin_slope_df["sum_norm_slope"] > 0) & (norm_range_mask)
        stop_candidate_mask = (filtered_bin_slope_df["sum_norm_slope"] < 0) & (norm_range_mask)
        start_candidates = filtered_bin_slope_df[start_candidate_mask]["bin"].unique().tolist()
        stop_candidates = filtered_bin_slope_df[stop_candidate_mask]["bin"].unique().tolist()

        start_site_bins.update(start_candidates if orientation == "+" else stop_candidates)
        stop_site_bins.update(stop_candidates if orientation == "+" else start_candidates)


    # Determine candidate transcripts using every combination of start/stop bins
    cartesian_site_bins = list(product(start_site_bins, stop_site_bins))

    data = [{"id": index, "start": bins[0], "end":bins[1]} for index, bins in enumerate(cartesian_site_bins)]
    possible_transcripts_df = create_possible_transcripts_df(reads_df, orientation, bin_width, data)

    # NOTE: possible_transcripts_df has starts larger than ends if on negative strand.

    #print("POSSIBLE TRANSCRIPTS DF")
    #print(possible_transcripts_df)

    if verbose:
        print("\t- Assigning reads to candidate transcripts")

    # Generate each individual transcript depth and model them with domains
    reads_shinking_df = reads_df.copy()
    depth_subtranscripts_df = pd.DataFrame(total_range_s, columns=["rel_position"])

    # This is to prevent a PerformanceWarning in pandas where columns are repeatedly inserted into
    # Source -> https://stackoverflow.com/a/75776863
    depth_subtranscripts_dict = {}

    # Assign reads to transcript candidates to calculate depth and train model on it
    # ? Can we use a reducing command to shrink the reads_df
    for row in possible_transcripts_df.itertuples():
        transcript_id = "T_{}".format(str(row.id))
        # Interior reads fall within the real start and end of a given transcript candidate
        interior_reads_mask = (reads_shinking_df["start"] >= row.real_end) & (reads_shinking_df["end"] <= row.real_start + bin_width)
        if orientation == "+":
            interior_reads_mask = (reads_shinking_df["start"] >= row.real_start) & (reads_shinking_df["end"] <= row.real_end + bin_width)
        interior_reads_df = reads_shinking_df[interior_reads_mask]
        # Shrink the list of remaining reads
        reads_shinking_df = reads_shinking_df[~(reads_shinking_df["name"].isin(interior_reads_df["name"]))]

        # Calculate base depth just among interior reads for a transcript
        depth_reads_df = pd.merge(depth_subtranscripts_df[["rel_position"]], interior_reads_df[["start", "end"]], how="cross")
        if depth_reads_df.empty:
            # No interior reads present for transcript
            continue
        depth_subtranscripts_dict[transcript_id] = pd.Series(get_raw_depth(depth_reads_df))

    temp_df = pd.DataFrame(data=depth_subtranscripts_dict.values(), index=depth_subtranscripts_dict.keys()).transpose()
    depth_subtranscripts_df = pd.concat([depth_subtranscripts_df, temp_df], axis="columns")

    if verbose:
        print("\t- Create linear models")

    linear_models_df = create_linear_models(possible_transcripts_df, depth_subtranscripts_df, bin_width, orientation)

    #print("LINEAR MODELS DF")
    #print(linear_models_df)

    # Initial condition of linear model
    linear_models_leftover_df = linear_models_df.copy()
    depth_frame_df = pd.DataFrame(total_range_s, columns=["pos"])
    depth_frame_df["depth"] = 0
    depth_frame_df["actual_depth"] = depth_df["raw_depth"]

    final_transcripts = dict()

    ### Add each linear model stepwise until additions of the linear models do not improve the score
    if verbose:
        print("\t- Predicting best candidate transcripts")
    while True:
        baseline_error = get_sum_of_squares(depth_frame_df["actual_depth"], depth_frame_df["depth"])
        data = [{"id":row.id, "a":row.a, "b":row.b, "dstart":row.dstart, "dend":row.dend} for row in linear_models_leftover_df.itertuples()]
        indiv_assessment_frame_df = pd.DataFrame(data, index=linear_models_leftover_df.index)
        indiv_assessment_frame_df["dstart"] = linear_models_leftover_df["dstart"] if orientation == "+" else linear_models_leftover_df["dend"]
        indiv_assessment_frame_df["dend"] = linear_models_leftover_df["dend"] if orientation == "+" else linear_models_leftover_df["dstart"]

        # Perform predictions for each transcript candidate
        for row in indiv_assessment_frame_df.itertuples():
            # ? could we use model.predict to get this instead?
            predicted_frame_df = predict_linear_domain(row, range_start, range_end)
            normalize_linear_domain_depth(depth_frame_df, predicted_frame_df)
            indiv_assessment_frame_df.loc[row.Index, "sum_error"] = get_sum_of_squares(depth_frame_df["actual_depth"], predicted_frame_df["predicted_depth"])
            indiv_assessment_frame_df.loc[row.Index, "depth_fraction"]  = len(predicted_frame_df[predicted_frame_df["depth"] > 0]) / len(predicted_frame_df)

        # I think the idea here is that if "sum_error" -> "u", "baseline_error" -> "v" and coefficient of determination (R^2) -> 1 - u/v
        # then a bad model would have a negative R^2 because the "sum_error" was worse than "baseline"

        # Normalize, then keep all transcript models that are better than the baseline in terms of their prediction
        indiv_assessment_frame_df["norm_error_old"] = baseline_error - indiv_assessment_frame_df["sum_error"]
        indiv_assessment_frame_df["norm_error"] = (baseline_error - indiv_assessment_frame_df["sum_error"]) * indiv_assessment_frame_df["depth_fraction"]
        indiv_assessment_frame_df = indiv_assessment_frame_df[indiv_assessment_frame_df["norm_error"] > 0]

        # If no more transcript models pass error cutoff, exit
        if indiv_assessment_frame_df.empty:
            break

        # Predict again on the best transcript model.
        # Use this predicted depth as new baseline error for the next iteration
        indiv_assessment_frame_df = indiv_assessment_frame_df.sort_values(by="norm_error", ascending=False)
        best_transcript = indiv_assessment_frame_df.iloc[0]["id"]
        best_leftover_transcript_s = linear_models_leftover_df[linear_models_leftover_df["id"] == best_transcript].squeeze()

        predicted_frame_df = predict_linear_domain(best_leftover_transcript_s, range_start, range_end)
        normalize_linear_domain_depth(depth_frame_df, predicted_frame_df)
        depth_frame_df["depth"] = predicted_frame_df["predicted_depth"]

        final_transcripts[best_transcript] = indiv_assessment_frame_df.iloc[0]["norm_error"]
        linear_models_leftover_df = linear_models_leftover_df[~(linear_models_leftover_df["id"] == best_transcript)]
        # If no more transcript candidates are left to process, break
        if linear_models_leftover_df.empty:
            break

    if verbose:
        print("\t- Finalizing transcript list")

    final_transcripts_df = possible_transcripts_df[possible_transcripts_df["id"].isin((list(final_transcripts.keys())))].copy()
    if not orientation == "+":
        # swap start/end values
        final_transcripts_df.loc[:, ["real_start", "real_end"]] = final_transcripts_df.loc[:, ["real_end", "real_start"]].values

    for row in final_transcripts_df.itertuples():
        chr = reads_df["chr"].unique()[0]
        operon_name = f"O_{region_index}_T{row.Index}"
        start = int(row.real_start)
        end = int(row.real_end)
        score = "."
        transcript_s = pd.Series([chr, start, end, operon_name, score, orientation], index=["chr", "start", "end", "name",  "score", "strand"])
        transcript_s["name"] = operon_name
        subtranscript_df.loc[len(subtranscript_df)] = transcript_s

        gff_s = pd.Series(index=GFF_OUTPUT_COLS)
        gff_s["seqid"] = chr
        gff_s["source"] = "rdoperon.py"
        gff_s["type"] = "transcript"
        gff_s["start"] = start + 1  # GFF3 are 1-indexed
        gff_s["end"] = end  # inclusive
        gff_s["score"] = round(final_transcripts[row.Index], 2)
        gff_s["strand"] = orientation
        gff_s["phase"] = "."
        gff_s["attributes"] = f"ID={operon_name}"
        annotation_df.loc[len(annotation_df)] = gff_s

    return subtranscript_df, annotation_df

def create_linear_models( possible_transcripts_df, depth_subtranscripts_df, bin_width, orientation):
    linear_models_df = pd.DataFrame(columns=["id", "a", "b", "dstart", "dend"])
    # Make linear models
    for row in possible_transcripts_df.itertuples():
        transcript_id = "T_{}".format(str(row.id))

        # Do not want transcripts with no reads assigned
        if transcript_id not in depth_subtranscripts_df.columns:
            continue

        # Filter regions where the bases are within the real start and end boundaries
        rel_position_mask = (row.real_start >= depth_subtranscripts_df["rel_position"]) & (depth_subtranscripts_df["rel_position"] >= row.real_end  + bin_width)
        if orientation == "+":
            rel_position_mask = (row.real_start <= depth_subtranscripts_df["rel_position"]) & (depth_subtranscripts_df["rel_position"] <= row.real_end + bin_width)

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
        linear_models_df.loc[row.Index, "dstart"] = row.real_start  # start larger than end if negative strand
        linear_models_df.loc[row.Index, "dend"] = row.real_end
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

def create_possible_transcripts_df(reads_df, orientation, bin_width, data):
    possible_transcripts_df = pd.DataFrame(data)
    possible_transcripts_df["length"] = possible_transcripts_df["start"] - possible_transcripts_df["end"]
    if orientation == "+":
        possible_transcripts_df["length"] = possible_transcripts_df["end"] - possible_transcripts_df["start"]

    # This is important. Need to sort be smallest to largest, so that reads can be assigned to transcripts appropriately
    possible_transcripts_df = possible_transcripts_df[possible_transcripts_df["length"] > 0].sort_values(by="length")
    possible_transcripts_df["real_start"] = reads_df["start"].min() + round(possible_transcripts_df["start"] * bin_width, 0)
    possible_transcripts_df["real_end"] = reads_df["start"].min() + round(possible_transcripts_df["end"] * bin_width, 0)
    return possible_transcripts_df

def create_slope_df(depth_df):
    slope_df = depth_df[["transcript", "rel_position", "bin", "norm_depth", "raw_depth"]].sort_values(by="rel_position")
    # Get the next position for the slope calculation (final row will be NaN)
    slope_df["next_raw_depth"] = slope_df["raw_depth"].shift(periods=-1)
    slope_df["next_norm_depth"] = slope_df["norm_depth"].shift(periods=-1)
    slope_df["norm_delta"] = (slope_df["next_raw_depth"]/10 * slope_df["next_norm_depth"]) - (slope_df["raw_depth"]/10 * slope_df["norm_depth"])
    slope_df["norm_split_depth"] = (slope_df["norm_depth"] + slope_df["next_norm_depth"]) / 2
    slope_df["norm_slope"] = slope_df["norm_delta"] / slope_df["norm_split_depth"]
    slope_df = slope_df.dropna()
    return slope_df

def create_depth_df(reads_df:pd.DataFrame, total_range_s:pd.Series, bin_labels:pd.Series):
    depth_df = pd.DataFrame(total_range_s, columns=["rel_position"])
    depth_df["transcript"] = reads_df["chr"].unique()[0]    # Since we grouped by chromosome, all reads should be in same chromosome
    for pos in depth_df["rel_position"].tolist():
        # ? Should this be left-inclusive
        depth_df.loc[depth_df["rel_position"] == pos, "raw_depth"] = len(reads_df[(reads_df["start"] <= pos) & (pos <= reads_df["end"])])

    depth_df["norm_depth"] = depth_df["raw_depth"] / max(depth_df["raw_depth"])
    depth_df["bin"] = bin_labels
    return depth_df

def get_raw_depth(depth_reads_df):
    """Return depth for this particular base (number of reads with this base in its interval)."""
    # ? Should this be left-inclusive
    depth_reads_df["in_range"] = depth_reads_df["rel_position"].between(depth_reads_df["start"], depth_reads_df["end"], inclusive="both")
    return depth_reads_df.groupby("rel_position")["in_range"].sum().to_list()  # to list, otherwise indexes don't match

#### Depth Evaluation Function
def predict_linear_domain(transcript_s, range_start, range_end):
    # row - dstart/dend - depth region
    # tstart/tend - transcript
    # row - a - coefficient
    # row - b - y-intercept

    data = [{"rel_position":i + range_start, "depth":0} for i in range(range_end - range_start)]
    depth_frame_df = pd.DataFrame(data)

    # NOTE: Explored using sckit-learn LinearRegression.predict, but it was slightly slower due to extra checks
    depth_frame_df["in_range"] = depth_frame_df["rel_position"].apply(is_in_range, args=(transcript_s.dstart, transcript_s.dend))
    in_range_df = depth_frame_df[depth_frame_df["in_range"]]

    # prediction is y = ax + b
    depth_frame_df.loc[depth_frame_df["in_range"], "depth"] = (in_range_df["rel_position"] * transcript_s.a) + transcript_s.b
    return depth_frame_df

def is_in_range(pos, start, end):
    """True if base position is in range, False otherwise."""
    # ? Should this be left-inclusive
    return start <= pos <= end

if __name__ == '__main__':
    main()
