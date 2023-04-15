#!/bin/bash
#SBATCH --job-name=FFS_MC3_500_250_L128_ID40_phi0.015
#SBATCH --time=71:59:59          # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
###SBATCH --mail-type=begin        # send email when job begins
###SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=ALL
#SBATCH --mail-user=polyachenkoya@princeton.edu
#SBATCH --output=slurm-FFS_MC3_500_250_L128_ID40_phi0.015.out
#SBATCH --constraint=cascade,skylake
#========================================
python run.py -mode FFS_AB      -L 128 -to_get_timeevol 0 -N_states_FFS 250 -N_init_states_FFS 500 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -to_show_on_screen 0 -my_seed 40 -N_runs 2

python run.py -mode FFS_AB_many -L 128 -to_get_timeevol 0 -N_states_FFS 250 -N_init_states_FFS 500 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16 -my_seed 1 -N_runs 100
