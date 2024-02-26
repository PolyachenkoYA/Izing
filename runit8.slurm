#!/bin/bash
#SBATCH --job-name=test8
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
###SBATCH --mail-type=begin        # send email when job begins
###SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=ALL
#SBATCH --mail-user=polyachenkoya@princeton.edu
###SBATCH --output=test2.out
#SBATCH --constraint=cascade,skylake
#========================================

module load gsl/2.6

#python run.py -mode FFS_AB -L 128 -to_get_timeevol 0 -N_states_FFS 10 -N_init_states_FFS 20 -N_runs 2 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16
#python run_tmp.py -mode FFS_AB_Tphi1 -Temp_s 0.95 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 50 -to_show_on_screen 0
#stdbuf -o0 -e0 python run_tmp.py -mode FFS_AB_Tphi1 -Temp_s 0.95 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 50 -to_show_on_screen 0
#python run.py -mode FFS_AB_Tphi1 -Temp_s 0.9 -phi1_s 0.015 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 15 20 30 50 100
#python run_tmp2.py -mode FFS_AB -L 400 -OP_interfaces_set_IDs mu10 -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -init_composition 0.09789 0.0111 0.01 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -to_recomp 0 -Dtop_Nruns 300 -my_seeds 1001
#python run_tmp2.py -mode BF_AB -Nt 100000000 -L 450 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.99 0.01 0.0 -e -2.68010292 -1.34005146 -1.71526587 -timeevol_stride 2000 -to_keep_composition 1 -to_recomp 10  -to_plot 0
#python run_tmp.py -mode BF_AB -Nt 880000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 0 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.9896 0.0104 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -timeevol_stride 2000 -to_recomp 10 -verbose 1 -to_plot 0 -my_seeds 24
#python run_tmp.py -mode BF_AB -Nt 300000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 0 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.0104 0.001 -e -2.68010292 -2.68010292 -2.68010292 -OP_0 2 -OP_max 150 -timeevol_stride 2000 -to_recomp 0 -verbose 1 -to_plot 0
python run_tmp2.py -mode BF_AB -Nt 1000000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0  -e -2.680103 -1.340051 -1.715266 -MC_move_mode swap -init_composition 0.0104 0.0 -OP_0 2 -OP_max 150 -timeevol_stride 1000 -R_clust_init 0 -to_recomp 0 -verbose 1 -to_plot 0
