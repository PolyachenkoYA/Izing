#import os
import subprocess
l = range(1, 5 + 1)
subprocess.run(''.join([('sbatch runit%d.slurm; ' % i) for i in l]), shell=True)
