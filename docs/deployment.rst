Deployment
==========

On the `Maxwell` cluster:

.. code-block:: bash

    git clone https://git.xfel.eu/gitlab/dataAnalysis/karaboFAI.git
    cd karaboFAI
    source /gpfs/exfel/sw/software/modules
    module load xfel
    pip install .

.. note::
    We have another deployment which serves as a fallback solution.
    **karaboFAI** will be deployed in both ways.

.. code-block:: bash

   cd /gpfs/exfel/sw/software
   rm -rf karaboFAI
   git clone https://git.xfel.eu/gitlab/dataAnalysis/karaboFAI.git
   cd karaboFAI
   module load anaconda3
   ./deploy.sh

.. note::
    We install the software in ``/gpfs/exfel/sw/software`` from the
    `Maxwell` cluster, where we have convenient Internet access. This
    folder is synchronised with ``/gpfs/exfel/sw/software`` on the online
    clusters, from where the tool will be used.
