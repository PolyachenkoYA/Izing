#import os
import subprocess
l = range(1, 13 + 1)
subprocess.run(''.join([('sbatch runit%d.slurm; ' % i) for i in l]), shell=True)
