#!/bin/bash
#SBATCH -A dhawals1939
#SBATCH -n 36
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=op_file_dhawal.txt

scp -r dhawals1939@ada.iiit.ac.in:/share3/dhawals1939/Amit_Pandey /scratch
python3 WMD_min_RE_for_loop_test50_Dhawal_scratch_v3.py