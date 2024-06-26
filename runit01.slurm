#!/bin/bash
#SBATCH --job-name=test01
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=200G         # memory per cpu-core (4G is default)
###SBATCH --mail-type=begin        # send email when job begins
###SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=ALL
#SBATCH --mail-user=polyachenkoya@princeton.edu
###SBATCH --output=test0.out
#SBATCH --constraint=cascade,skylake
#========================================

module purge
module load anaconda3/2024.2
module load intel/2021.1.2
module load intel-mpi/intel/2021.1.1
module load fftw/intel-2021.1/intel-mpi/3.3.9

conda activate fast-mpi4py

#python run.py -mode FFS_AB -L 128 -to_get_timeevol 0 -N_states_FFS 5 -N_init_states_FFS 10 -N_runs 2 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -init_composition 0.975 0.015 0.01 -OP_interfaces_set_IDs nvt16
#python run_tmp.py -mode FFS_AB_Tphi1 -Temp_s 0.95 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 50 -to_show_on_screen 0
#stdbuf -o0 -e0 python run_tmp.py -mode FFS_AB_Tphi1 -Temp_s 0.85 -phi1_s 0.014 0.0145 0.015 0.0155 0.016 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 50 -to_show_on_screen 0
#python run.py -mode FFS_AB_Tphi1 -Temp_s 0.9 -phi1_s 0.0145 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 15 20 30 50 100
#python run_tmp2.py -mode FFS_AB_Tphi1 -Temp_s 1.0 0.95 0.9 -phi1_s 0.014 0.0145 0.015 0.0155 -L 128 -to_get_timeevol 0 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -phi2 0.01 -OP_interfaces_set_IDs nvt -to_recomp 2 -font_mode present -OP0_constr_s 15 20 30 50 100 -to_do_Dtop 1 -N_ID_groups 1
#python run_tmp2.py -mode FFS_AB_many -L 64 -OP_interfaces_set_IDs mu8 -to_get_timeevol 0 -N_states_FFS 300 -N_init_states_FFS 600 -mu 5.25 4.92238326 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode flip -to_recomp 0 -Dtop_Nruns 30000 -my_seeds 1000 1001 1002 1003 1004
#python run_tmp2.py -mode FFS_AB -L 400 -OP_interfaces_set_IDs mu10 -to_get_timeevol 0 -N_states_FFS 30 -N_init_states_FFS 60 -init_composition 0.09789 0.0111 0.01 -e -2.68010292 -1.34005146 -1.71526587 -MC_move_mode long_swap -to_recomp 0 -Dtop_Nruns 300 -my_seeds 1000
#python run_tmp.py -mode BF_AB -Nt 100000000 -L 450 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.99 0.01 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -timeevol_stride 2000 -R_clust_init 8.17 -to_keep_composition 1 -to_recomp 10
#python run_tmp.py -mode BF_AB -Nt 880000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 0 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.9896 0.0104 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -timeevol_stride 2000 -to_recomp 10 -verbose 1 -to_plot 0
#python run_tmp.py -mode BF_AB -Nt 225000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 0 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.0104 0.019206 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -OP_max 150 -timeevol_stride 2000 -to_recomp 0 -verbose 1 -to_plot 0
#python run_tmp.py -mode BF_AB -Nt 200000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.9896 0.0104 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -OP_max 150 -timeevol_stride 2000 -to_recomp 0 -verbose 1 -my_seeds 1
#python run_tmp.py -mode BF_AB -Nt 300000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.0105 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 2 -OP_max 150 -timeevol_stride 2000 -to_recomp 0 -verbose 1
#python run_tmp2.py -mode BF_1 -Nt 6000000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 0 -N_saved_states_max 0 -MC_move_mode long_swap -init_composition 0.0101 0.0 -e -2.68010292 -1.34005146 -1.71526587 -OP_0 25 -OP_max 150 -timeevol_stride 2000 -to_recomp 0 -verbose 1 -to_plot 0 -my_seeds 23
#python run_tmp3.py -mode BF_AB -Nt 300000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 50010 -e -2.680103 -1.340051 -1.715266 -MC_move_mode long_swap -init_composition 0.010 0.0 -OP_0 5 -OP_max 150 -timeevol_stride 1000 -R_clust_init 0 -to_recomp 0 -BF_hist_edges_ID mu3 -my_seeds 24  -progress_print_stride -1000
#python run_tmp2.py -mode BF_AB_many -Nt 3300000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 33010 -e -2.680103 -1.340051 -1.715266 -MC_move_mode swap -init_composition 0.0104 0.0 -OP_0 5 -OP_max 150 -timeevol_stride 100000 -R_clust_init 0 -to_recomp postproc_hard -BF_hist_edges_ID mu3 -progress_print_stride -10000 -font_mode present -my_seeds 25 26 27 28 29 30 31 32 33 34 35 36
python run_tmp1.py -mode BF_AB_many -Nt 3300000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 33010 -e -2.680103 -1.340051 -1.715266 -MC_move_mode swap -init_composition 0.0104 0.0 -OP_0 5 -OP_max 150 -timeevol_stride 100000 -R_clust_init 0 -to_recomp 0             -BF_hist_edges_ID mu3 -progress_print_stride -10000 -font_mode present -my_seeds 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
#python run_tmp2.py -mode BF_AB -Nt 3300000000 -L 300 -to_get_timeevol 1 -to_plot_timeevol 1 -N_saved_states_max 33010 -e -2.680103 -1.340051 -1.715266 -MC_move_mode swap -init_composition 0.0104 0.0 -OP_0 5 -OP_max 150 -timeevol_stride 100000 -R_clust_init 0 -to_recomp postproc_hard -BF_hist_edges_ID mu3 -progress_print_stride -10000 -font_mode present -my_seeds 24
