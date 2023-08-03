#!/bin/bash
#SBATCH --job-name=big1
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=50G         # memory per cpu-core (4G is default)
###SBATCH --mail-type=begin        # send email when job begins
###SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=ALL
#SBATCH --mail-user=polyachenkoya@princeton.edu
#SBATCH --output=big1.out
#SBATCH --constraint=cascade,skylake
#========================================
#python run_tmp2.py -mode FFS_AB -L 320 -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -e -2.680103 -1.340051 -1.715266 -MC_move_mode long_swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvtBig -to_show_on_screen 0 -my_seeds 1000
#python run_tmp2.py -mode FFS_AB -L 320 -OP_interfaces_set_IDs mu5 -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -mu 5.15 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode flip -to_recomp 2 -Dtop_Nruns 300 -my_seeds 1000
#python run.py -mode FFS_AB -L 320 -OP_interfaces_set_IDs nvtBig -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -mu 5.15 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode flip -to_recomp 3 -Dtop_Nruns 1000
python run_tmp2.py -mode FFS_AB -L 400 -OP_interfaces_set_IDs mu10 -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -init_composition 0.09789 0.0111 0.01 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -to_recomp 0 -Dtop_Nruns 300 -my_seeds 1000
