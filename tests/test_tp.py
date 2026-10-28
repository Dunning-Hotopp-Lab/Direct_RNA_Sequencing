"""Regression tests for tp.py using the small E. coli E2348/69 plasmid dataset in test-data/"""

import subprocess, sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
TP = REPO_DIR / "tp.py"
TEST_DATA = REPO_DIR / "test-data"
EXPECTED = TEST_DATA / "expected"
READS = TEST_DATA / "E2348_69_plasmids.reads.bed"
RANGES = TEST_DATA / "E2348_69_plasmids.ranges.bed"

def run_tp(*args, cwd):
    return subprocess.run([sys.executable, str(TP), *map(str, args)], cwd=cwd, capture_output=True, text=True)

def test_version():
    result = run_tp("--version", cwd=REPO_DIR)
    assert result.returncode == 0
    assert "1.0.5" in result.stdout

def test_defaults_match_expected(tmp_path):
    result = run_tp("-r", READS, "-d", RANGES, "-o", "out", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    for suffix in ("candidates.bed", "bed", "gff3"):
        expected = (EXPECTED / f"E2348_69_plasmids.{suffix}").read_text()
        assert (tmp_path / f"out.{suffix}").read_text() == expected, suffix

def test_gff3_header(tmp_path):
    run_tp("-r", READS, "-d", RANGES, "-o", "out", cwd=tmp_path)
    lines = (tmp_path / "out.gff3").read_text().splitlines()
    assert lines[0] == "##gff-version 3"
    assert all(line.split("\t")[1] == "tp.py" for line in lines[1:])

def test_candidates_only_with_chromosome(tmp_path):
    result = run_tp("-r", READS, "-d", RANGES, "-c", "NZ_CP059842.2", "--candidates_only", "-o", "out", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    expected = (EXPECTED / "E2348_69_plasmids.NZ_CP059842.2.candidates.bed").read_text()
    assert (tmp_path / "out.candidates.bed").read_text() == expected
    assert not (tmp_path / "out.bed").exists()
    assert not (tmp_path / "out.gff3").exists()

def test_max_depth_skips_regions(tmp_path):
    result = run_tp("-r", READS, "-d", RANGES, "--max_depth", "200", "-o", "out", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "depth is over the max threshold of 200" in result.stdout
    for suffix in ("candidates.bed", "bed", "gff3"):
        expected = (EXPECTED / f"E2348_69_plasmids.max_depth_200.{suffix}").read_text()
        assert (tmp_path / f"out.{suffix}").read_text() == expected, suffix

def test_numeric_chromosome_ids(tmp_path):
    # Numeric chromosome IDs must still match --chromosome
    rename = {"NZ_CP059841.1": "1", "NZ_CP059842.2": "2"}
    for src in (READS, RANGES):
        lines = [line.split("\t", 1) for line in src.read_text().splitlines()]
        (tmp_path / src.name).write_text("".join(f"{rename[c]}\t{rest}\n" for c, rest in lines))
    result = run_tp("-r", READS.name, "-d", RANGES.name, "-c", "1", "-o", "out", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    out_chrs = {line.split("\t")[0] for line in (tmp_path / "out.bed").read_text().splitlines()}
    assert out_chrs == {"1"}

def test_header_lines_are_skipped(tmp_path):
    reads = tmp_path / "reads.bed"
    reads.write_text("track name=reads\n#comment\n" + READS.read_text())
    result = run_tp("-r", reads, "-d", RANGES, "-o", "out", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out.bed").read_text() == (EXPECTED / "E2348_69_plasmids.bed").read_text()

@pytest.mark.parametrize("bad_args", [
    ["--assigned_read_ratio_threshold", "1.5"],
    ["--min_depth", "-1"],
    ["--max_depth", "0"],
])
def test_invalid_thresholds_exit_nonzero(tmp_path, bad_args):
    result = run_tp("-r", READS, "-d", RANGES, *bad_args, cwd=tmp_path)
    assert result.returncode != 0

def test_empty_ranges_file_exits_nonzero(tmp_path):
    empty = tmp_path / "empty.bed"
    empty.write_text("")
    result = run_tp("-r", READS, "-d", empty, cwd=tmp_path)
    assert result.returncode != 0
    assert "No data lines" in result.stderr
