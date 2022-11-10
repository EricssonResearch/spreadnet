SBATCH on Uppmax
--------------------

* Submit batch to Uppmax::

    sbatch [filename]

* Check job status (paste exactly)::

    jobinfo -u "$(whoami)" -M snowy


* To check logs, look at `.out` files in this folder matching your job id. Or list them by date by using::

    ls -l

    less slurm-[job_id].out

    :q
