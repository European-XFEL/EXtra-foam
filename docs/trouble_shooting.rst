TROUBLE SHOOTING
================

.. _FastX2: https://confluence.desy.de/display/IS/FastX2
.. _max-display: https://max-display.desy.de:3443/

Could not connect to display
++++++++++++++++++++++++++++

While trying to run **EXtra-foam** remotely on the online cluster (exflonc12, etc), if you
end up with error messages similar to,

.. code-block:: console

   qt.qpa.xcb: could not connect to display
   qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
   This application failed to start because no Qt platform plugin could be initialized.
   Reinstalling the application may fix this problem.

please make sure that you have done X11 forwarding while logging to the online cluster.
Using **EXtra-foam** on Maxwell cluster, it is better to use FastX2_ at max-display_ as
explained in previous section.

- **Shut down the redis server?**

If you are prompted to warnings like,

.. code-block:: console

   [user@exflonc12 ~]$ extra-foam DSSC SCS

   services.py - WARNING - Found Redis server for DSSC (started at 2020-02-06 12:50:03.906872)
   already running on this machine using port 6380!

   You can choose to shut down the Redis server. Please note that the owner of the Redis server
   will be informed (your username and IP address).

   Shut down the existing Redis server? (y/n)

**EXtra-foam** uses `Redis` as broker to pass meta information between different processes. By
design, each type of detector has its unique `Redis` port so one can safely run more than one
**EXtra-foam** instances for different detectors on the same machine. However, it is not allowed
to run two instances with the same type of detector. Also, **EXtra-foam** receives data from
**karabo bridge** and thus there can be data loss if there is any instance secretly running
in the background, stealing the data.

In the instrument control room, there should be only one **EXtra-foam** instance for the detector
that is running. Therefore, it is safe to type "y" to shut down the existing *Redis* server.
However, if somebody wants to make a joke about you and did that remotely, you will get informed.

Config file is invalid
++++++++++++++++++++++

If you are prompted to warning like,

.. code-block:: console

   Traceback (most recent call last):
     File "/home/username/anaconda3/envs/foam/bin/extra-foam", line 11, in <module>
       load_entry_point('EXtra-foam', 'console_scripts', 'extra-foam')()
     File "/home/username/xfel-data-analyais/EXtra-foam/extra_foam/services.py", line 356, in application
       config.load(detector, topic)
     File "/home/username/xfel-data-analyais/EXtra-foam/extra_foam/config.py", line 456, in load
       self._data.load(detector, topic)
     File "/home/username/xfel-data-analyais/EXtra-foam/extra_foam/config.py", line 382, in load
       self.from_file(det, topic)
     File "/home/username/xfel-data-analyais/EXtra-foam/extra_foam/config.py", line 393, in from_file
       raise OSError(msg)
   OSError: Invalid config file: /home/username/.EXtra-foam/scs.config.yaml
   ParserError('while parsing a block mapping', <yaml.error.Mark object at 0x7fcffbd84910>,
   "expected <block end>, but found '<block mapping start>'", <yaml.error.Mark object at 0x7fcffbd84ed0>)

This error is triggered when the :ref:`config file` is not valid. Please correct it if you have modified
the default one. Alternatively, you can delete it and let the program generate a default one for you.

No data is received
+++++++++++++++++++

If **EXtra-foam** opens up fine and running it by clicking on *start* button does
nothing, please make sure that relevant **PipeToZeroMQ** device is properly
configured, activated and its *data sent* property is updating. This device
can be configured only with the help of experts (data analysis support and beamline scientists).

Incorrect dependencies
++++++++++++++++++++++

If you happen to have some Python packages which are used in **EXtra-foam** but
in the meanwhile installed in your `~/.local` directory, anything bad can happen if
they have different version numbers. In the best case, **EXtra-foam** crashes, for example,
when it attempted to call a function which does not exist in your local installation, and you
will immediately notice it. In the worst case, the software keeps running but the result
is incorrect and the difference between the incorrect result and the correct one is
unperceivable.

This is the downside of using Anaconda to deploy software. However, it is easy to
track down the loathful Python package if you already have one or a few suspects. Assuming
the module `EXtra-foam/beta` is loaded, you can check the suspect one-by-one by following
the example below:

.. code-block:: console

  ~ python
  Python 3.7.3 (default, Mar 27 2019, 22:11:17)
  [GCC 7.3.0] :: Anaconda, Inc. on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import redis
  >>> print(redis.__file__)
  # expected result :-)
  /gpfs/exfel/sw/software/xfel_anaconda3/EXtra-foam-beta/lib/python3.7/site-packages/redis/__init__.py
  # This is bad!
  /home/username/.local/lib/python3.7/site-packages/redis/__init__.py

The remedy is simply. Run `pip uninstall` to remove your local installation.
