"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
from collections.abc import MutableSet


class Stack:
    """An LIFO stack."""
    def __init__(self):
        self.__items = []

    def push(self, item):
        """Append a new element."""
        self.__items.append(item)

    def pop(self):
        """Return and remove the top element."""
        return self.__items.pop()

    def top(self):
        """Return the first element."""
        if self.empty():
            raise IndexError("Stack is empty")

        return self.__items[-1]

    def empty(self):
        return not self.__items

    def __len__(self):
        return len(self.__items)


class OrderedSet(MutableSet):
    def __init__(self, sequence=None):
        super().__init__()

        if sequence is None:
            self._data = OrderedDict()
        else:
            kwargs = {v: 1 for v in sequence}
            self._data = OrderedDict(**kwargs)

    def __contains__(self, item):
        """Override."""
        return self._data.__contains__(item)

    def __iter__(self):
        """Override."""
        return self._data.__iter__()

    def __len__(self):
        """Override."""
        return self._data.__len__()

    def add(self, item):
        """Override."""
        self._data.__setitem__(item, 1)

    def discard(self, item):
        """Override."""
        if item in self._data:
            self._data.__delitem__(item)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self._data.keys())})"
