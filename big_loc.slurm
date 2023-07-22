#!/bin/bash
#SBATCH --job-name=FFS_MC2_30_15_L320_Temp1.0_ID1000_phi0.015_OPsnvtBig
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
###SBATCH --mail-type=begin        # send email when job begins
###SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=ALL
#SBATCH --mail-user=polyachenkoya@princeton.edu
#SBATCH --output=/scratch/gpfs/yp1065/Izing/logs/FFS_MC3_60_30_L320_Temp1.0_ID1000_phi0.015_OPsnvtBig.out
#SBATCH --constraint=cascade,skylake
#========================================
python run_tmp2.py -mode FFS_AB -L 320 -to_get_timeevol 0 -N_states_FFS 15 -N_init_states_FFS 30 -e -2.680103 -1.340051 -1.715266 -Temp 1.0 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvtBig -to_show_on_screen 0 -my_seeds 1000 