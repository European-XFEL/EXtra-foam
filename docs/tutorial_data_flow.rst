DATA FLOW
=========

.. _Karabo: https://doi.org/10.1107/S1600577519006696
.. _karabo-bridge-py: https://github.com/European-XFEL/karabo-bridge-py
.. _ZeroMQ: https://github.com/zeromq

..


At European XFEL, **EXtra-foam** receives data from the distributed control framework Karabo_.

Data received from different sensors/detectors located at different locations need to to be
"aligned" for further analysis. At European XFEL, the accelerator produces 10 bunch trains per
second. As a result, data are stamped and can only be aligned by **Train ID**. For big modular
detectors like AGIPD, LPD, DSSC, etc., alignment of data from different modules are carried out
in the so-called "calibration pipeline". However, in most use cases, users will also want to align
the modular detector data with motor positions and/or some other detectors (e.g. XGM, digitizer).
Furthermore, motors can only produce data at rate of 2 Hz, which makes the alignment even more
difficult in real-time. Fortunately, the aforementioned alignment is taken care of by
*DataCorrelator*/*TrainMatcher* devices at European XFEL and users do not need to worry about that.

After the data from different sources are aligned, they will be serialized and streamed using
ZeroMQ_. This process is performed inside the *PipeToZeroMQ* device, which is also known as
"Karabo bridge". **EXtra-foam** makes use of the karabo-bridge-py_, which is the client of the
"Karabo bridge", to receive and deserialize the data.

.. image:: images/data_flow.png
   :width: 800