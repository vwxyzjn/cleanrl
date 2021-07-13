import sys
import os
import conda.cli
is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))

if is_conda:
    conda_env = os.environ['CONDA_DEFAULT_ENV']
    conda.cli.main("conda", "env", "export")