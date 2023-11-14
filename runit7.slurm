#!/bin/bash
#SBATCH --job-name=test7
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
###SBATCH --mail-type=begin        # send email when job begins
###SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=ALL
#SBATCH --mail-user=polyachenkoya@princeton.edu
###SBATCH --output=test1.out
#SBATCH --constraint=cascade,skylake
#========================================

#python run.py -mode FFS_AB -L 128 -to_get_timeevol 0 -N_states_FFS 5 -N_init_states_FFS 10 -N_runs 2 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16
#python run_tmp.py -mode FFS_AB_Tphi1 -Temp_s 0.95 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 50 -to_show_on_screen 0
#stdbuf -o0 -e0 python run_tmp.py -mode FFS_AB_Tphi1 -Temp_s 0.85 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 50 -to_show_on_screen 0
#python run.py -mode FFS_AB_Tphi1 -Temp_s 0.9 -phi1_s 0.0145 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 15 20 30 50 100
#python run_tmp2.py -mode FFS_AB_Tphi1 -Temp_s 1.0 0.95 0.9 -phi1_s 0.014 0.0145 0.015 0.0155 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 15 20 30 50 100 -to_do_Dtop 1 -N_ID_groups 1
#python run_tmp2.py -mode FFS_AB_many -L 64 -OP_interfaces_set_IDs mu8 -to_get_timeevol 0 -N_states_FFS 300 -N_init_states_FFS 600 -mu 5.25 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode flip -to_recomp 0 -Dtop_Nruns 30000 -my_seeds 1000 1001 1002 1003 1004
#python run_tmp2.py -mode FFS_AB -L 400 -OP_interfaces_set_IDs mu10 -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -init_composition 0.09789 0.0111 0.01 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -to_recomp 0 -Dtop_Nruns 300 -my_seeds 1000
#python run_tmp.py -mode BF_AB -Nt 100000000 -L 450 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.99 0.01 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -timeevol_stride 2000 -R_clust_init 8.17 -to_keep_composition 1 -to_recomp 10
#python run_tmp.py -mode BF_AB -Nt 880000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 0 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.9896 0.0104 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -timeevol_stride 2000 -to_recomp 10 -verbose 1 -to_plot 0
python run_tmp.py -mode BF_AB -Nt 300000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 0 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.0104 0.00010 -e -2.68010292 -2.68010292 -2.68010292 -OP_0 2 -OP_max 150 -timeevol_stride 2000 -to_recomp 0 -verbose 1 -to_plot 0
