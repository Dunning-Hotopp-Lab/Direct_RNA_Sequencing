# Test data

Small subset of the *E. coli* E2348/69 direct RNA sequencing data from the tp.py manuscript
(Mattick et al. 2024, mBio, https://doi.org/10.1128/mbio.02359-24; SRA: SRR18061003, SRR18061004),
restricted to 11 ranges on the two plasmids (NZ_CP059841.1 and NZ_CP059842.2) of GCF_014117345.2.

## Inputs

* `E2348_69_plasmids.reads.bed` - reads (BED6) falling entirely within the ranges below, on the same strand.
  Generated from the merged alignments with `bamToBed -i merged_all.bam > merged_all.readtable.bed`
* `E2348_69_plasmids.ranges.bed` - distinct ranges per strand (chr, start, end, strand, read count).
  Generated with `bedtools merge -i merged_all.readtable.bed -s -c 6,4 -o distinct,count | awk '$5>20 {print $0}'`

See https://github.com/jdhotopp/tp.py-Direct-RNA-Sequencing-Manuscript- for the full manuscript commands.

## Expected outputs

```
tp.py -r E2348_69_plasmids.reads.bed -d E2348_69_plasmids.ranges.bed -o expected/E2348_69_plasmids
tp.py -r E2348_69_plasmids.reads.bed -d E2348_69_plasmids.ranges.bed -c NZ_CP059842.2 --candidates_only -o expected/E2348_69_plasmids.NZ_CP059842.2
tp.py -r E2348_69_plasmids.reads.bed -d E2348_69_plasmids.ranges.bed --max_depth 200 -o expected/E2348_69_plasmids.max_depth_200
```
