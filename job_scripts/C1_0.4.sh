#!/bin/bash
#$-l rt_G.small=1
#$-l h_rt=144:00:00
#$-j y
#$-cwd
source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8/3.8.7 cuda/11.0/11.0.3 cudnn/8.0/8.0.5
source ~/env_qh/bin/activate
cd /home/acc13085dy/QH_ga/C1_0.4/GA
python main.py