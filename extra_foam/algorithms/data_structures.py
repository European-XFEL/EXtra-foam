"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""


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
