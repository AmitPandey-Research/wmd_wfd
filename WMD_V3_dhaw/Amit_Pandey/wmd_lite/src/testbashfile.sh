#!/bin/bash
#SBATCH -A dhawals1939
#SBATCH -n 30
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=op_file.txt
python3 wmdbash_v2.py
