#!/bin/bash
#SBATCH --time      0-48:00:00
#SBATCH --nodes     1
#SBATCH --partition allgpu
#SBATCH --job-name  lgatr
#SBATCH --output    job_logs/out_toprecogatr.out
#SBATCH --error     job_logs/err_toprecogatr.err
export LD_PRELOAD=""                 # useful on max-display nodes, harmless on others
source /etc/profile.d/modules.sh     # make the module command available
module load cuda/11.8
module load maxwell mamba
module load maxwell texlive/2022
. mamba-init
mamba activate lgatr
echo "starting lgatr"
cd /data/dust/user/bachjoer/beegfs.migration/ml-workspace/lgatr
python3 run.py --config-name=top_reco 
