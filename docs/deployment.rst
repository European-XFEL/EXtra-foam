Deployment
==========

On the `Maxwell` cluster:

.. code-block:: bash

   cd /gpfs/exfel/sw/software
   git clone https://git.xfel.eu/gitlab/dataAnalysis/karaboFAI.git
   cd karaboFAI
   module load anaconda3
   ./deploy.sh

.. note::
    We install the software in ``/gpfs/exfel/sw/software`` from the
    `Maxwell` cluster, where we have convenient Internet access. This
    folder is synchronised with ``/gpfs/exfel/sw/software`` on the online
    clusters, from where the tool will be used.
