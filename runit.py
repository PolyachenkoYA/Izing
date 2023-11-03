#import os
import subprocess
l = range(7, 12 + 1)
subprocess.run(''.join([('sbatch runit%d.slurm; ' % i) for i in l]), shell=True)
