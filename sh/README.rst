SBATCH on Uppmax
--------------------

* Submit job batch to Uppmax (does not require shelling into singularity)::

    sbatch [filename]

* Check job status (paste exactly)::

    jobinfo -u "$(whoami)" -M snowy

* To cancel jobs::

    scancel -u "$(whoami)" -M snowy [job_id]

* To check logs, look at `.out` files in this folder matching your job id. Or list them by date by using::

    ls -l
    tail -f slurm-[job_id].out

    or

    less slurm-[job_id].out

    :q

* For more guides, please visit https://www.uppmax.uu.se/support/user-guides/slurm-user-guide/#tocjump_025102109739779_4
