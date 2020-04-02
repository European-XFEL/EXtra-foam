"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import abc
import functools

from PyQt5.QtWidgets import QFrame

from ..mediator import Mediator


def create_imagetool_view(*klasses, **kw_klasses):
    """A decorator for views in ImageTool."""
    def wrap(instance_type):
        @functools.wraps(instance_type)
        def wrapped_instance_type(*args, **kwargs):
            if klasses:
                if len(args) > 1:
                    raise ValueError(
                        "Only one non-keyword ctrl widget class is allowed!")
                instance_type._ctrl_instance_type = klasses[0]
            elif kw_klasses:
                instance_type._ctrl_instance_type = []
                for name, kls in kw_klasses.items():
                    instance_type._ctrl_instance_type.append((name, kls))
            else:
                raise ValueError("At least one ctrl widget class is required!")

            return instance_type(*args, **kwargs)
        return wrapped_instance_type
    return wrap


class _AbstractImageToolView(QFrame):
    """_AbstractImageToolView class.

    Base class for all the "views" in ImageToolWindow.
    """

    _ctrl_instance_type = None

    def __init__(self, *, pulse_resolved=True, parent=None):
        """Initialization.

        :param bool pulse_resolved: whether the related data is
            pulse-resolved or not.
        """
        super().__init__(parent=parent)

        self._pulse_resolved = pulse_resolved

        self._mediator = Mediator()

        if self._ctrl_instance_type is not None:
            if isinstance(self._ctrl_instance_type, list):
                for name, instance_type in self._ctrl_instance_type:
                    self.__dict__[name] = self.parent().createCtrlWidget(
                        instance_type)
            else:
                self._ctrl_widget = self.parent().createCtrlWidget(
                    self._ctrl_instance_type)

    @abc.abstractmethod
    def initUI(self):
        """Initialization of UI."""
        raise NotImplementedError

    @abc.abstractmethod
    def initConnections(self):
        """Initialization of signal-slot connections."""
        raise NotImplementedError

    def updateF(self, data, auto_update):
        """This method is called by ImageTool."""
        # Views in ImageTool are not registered and updated blindly.
        raise NotImplementedError

    def onActivated(self):
        """Handler when the view is activated."""
        pass

    def onDeactivated(self):
        """Handler when the view is deactivated."""
        pass
