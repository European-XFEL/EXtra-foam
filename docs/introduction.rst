INTRODUCTION
============

**EXtra-foam** ( **F**\ ast **O**\ nline **A**\ nalysis **M**\ onitor) is a tool that provides
real-time and off-line data analysis (**azimuthal integration**, **ROI**, **correlation**,
**binning**, etc.) and visualization for experiments using **2D area detectors** (*AGIPD*,
*DSSC* , *FastCCD*, *JungFrau*, *LPD*, etc.) at European XFEL.


+------------+-------------------------+-------------------------+
|            | online                  | offline                 |
|            +------------+------------+------------+------------+
|            | raw        | calibrated | raw        | calibrated |
+============+============+============+============+============+
| AGIPD      | True       | True       | True       | True       |
+------------+------------+------------+------------+------------+
| LPD        | True       | True       | True       | True       |
+------------+------------+------------+------------+------------+
| DSSC       | True       | True       | True       | True       |
+------------+------------+------------+------------+------------+
| JungFrau   | False      | True       | True       | True       |
+------------+------------+------------+------------+------------+
| FastCCD    | True       | True       | True       | True       |
+------------+------------+------------+------------+------------+


Why use **EXtra-foam**
----------------------

1. It allows users to perform EDA (exploratory data analysis) in real time by 'probing'
the data with a combination of different analysis tools, for instance, monitoring individual
pulses in a train, checking correlation and trying different normalization methods, etc.
This is particularly useful if users are not sure what the data really look like or want to have
a sanity check;

2. It provides tailored data analysis configuration and visualization for specific experiments.
For example, in *pump-probe* setup, it allows users to choose how the pump and probe pulses
are distributed (e.g. in the same train or different train) by providing several typical "modes".
It also integrates important plots in a single window so that users can gather abundant information
in a glance;

3. It allows uses to **replay the experiment** with files. This is another very useful
feature since for both newcomers and veterans. For newcomers, it helps to understand what whey
will see during a real experiment by running the 'replay' with some sample/real data; for veterans,
it helps to optimize the parameters which in turn provides a better real-time monitoring and feedback
during experiments. Moreover, *When starting a new type of experiment, you may not be able to observe
the expected signal during the first run. It could be worthy of double checking the analysis setup by
replaying the experiment before searching for other reasons.* It is worth noting that the 'replay'
result could be different from the real-time result if you are using the real-time calibration service
from Karabo, since the offline calibration algorithms are more complicated than the real-time ones.


Performance
-----------

For train-resolved detectors like JungFrau and FastCCD, **EXtra-foam** can process and visualize the data
at **10 Hz**. For large detectors like AGIPD, LPD and DSSC, the processing and visualization rate is
**2 ~ 3 Hz** with 64 pulses/train. It can even process the DSSC data with **400 pulses/train** at a rate of
about 0.3 ~ 0.4 Hz.


Galleries
-----------

.. image:: images/MainGUI.png
   :width: 800

.. image:: images/ImageTool.png
   :width: 800

.. image:: images/pump-probe_window.png
   :width: 800

.. image:: images/1D_binning_window.png
   :width: 800

