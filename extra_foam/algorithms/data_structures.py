"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import abstractmethod
from collections import namedtuple, OrderedDict
from collections.abc import MutableSet, Sequence

import numpy as np


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


class _AbstractSequence(Sequence):
    """Abstract class for 'Sequence' data.

    It cannot be instantiated without subclassing.
    """
    _OVER_CAPACITY = 2

    def __init__(self, max_len=3000):
        self._max_len = max_len

        self._i0 = 0  # starting index
        self._len = 0

    def __len__(self):
        """Override."""
        return self._len

    def capacity(self):
        return self._max_len

    @abstractmethod
    def data(self):
        """Return all the data."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the data history."""
        pass

    @abstractmethod
    def append(self, item):
        """Add a new data point."""
        pass

    @abstractmethod
    def extend(self, items):
        """Add a list of data points."""
        pass

    @classmethod
    def from_array(cls, *args, **kwargs):
        """Construct from array(s)."""
        raise NotImplementedError


class SimpleSequence(_AbstractSequence):
    """Store the history of scalar data."""

    def __init__(self, *, max_len=100000, dtype=np.float64):
        super().__init__(max_len=max_len)

        self._dtype = dtype
        self.resize(max_len)

    def __getitem__(self, index):
        """Override."""
        return self._x[self._i0:self._i0 + self._len][index]

    def data(self):
        """Override."""
        return self._x[slice(self._i0, self._i0 + self._len)]

    def append(self, item):
        """Override."""
        self._x[self._i0 + self._len] = item
        max_len = self._max_len
        if self._len < max_len:
            self._len += 1
        else:
            self._i0 += 1
            if self._i0 == max_len:
                self._i0 = 0
                self._x[:max_len] = self._x[max_len:]

    def extend(self, items):
        for item in items:
            self.append(item)

    def resize(self, max_len):
        self._max_len = max_len
        self._x = np.zeros(self._OVER_CAPACITY * max_len, dtype=self._dtype)
        self.reset()

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)

    @classmethod
    def from_array(cls, ax, *args, **kwargs):
        instance = cls(*args, **kwargs)
        for x in ax:
            instance.append(x)
        return instance


class SimpleVectorSequence(_AbstractSequence):
    """Store the history of vector data."""

    def __init__(self, size, *, max_len=100000, dtype=np.float64, order='C'):
        super().__init__(max_len=max_len)

        self._x = np.zeros((self._OVER_CAPACITY * max_len, size),
                           dtype=dtype, order=order)
        self._size = size

    @property
    def size(self):
        return self._size

    def __getitem__(self, index):
        """Override."""
        return self._x[self._i0:self._i0 + self._len, :][index]

    def data(self):
        """Override."""
        return self._x[slice(self._i0, self._i0 + self._len), :]

    def append(self, item):
        """Override.

        :raises: ValueError, if item has different size;
                 TypeError, if item has no method __len__.
        """
        if len(item) != self._size:
            raise ValueError(f"Item size {len(item)} differs from the vector "
                             f"size {self._size}!")

        self._x[self._i0 + self._len, :] = np.array(item)
        max_len = self._max_len
        if self._len < max_len:
            self._len += 1
        else:
            self._i0 += 1
            if self._i0 == max_len:
                self._i0 = 0
                self._x[:max_len, :] = self._x[max_len:, :]

    def extend(self, items):
        """Override."""
        for item in items:
            self.append(item)

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)

    @classmethod
    def from_array(cls, ax, *args, **kwargs):
        instance = cls(*args, **kwargs)
        for x in ax:
            instance.append(x)
        return instance


class SimplePairSequence(_AbstractSequence):
    """Store the history a pair of scalar data.

    Each data point is pair of data: (x, y).
    """

    def __init__(self, *, max_len=3000, dtype=np.float64):
        super().__init__(max_len=max_len)
        self._x = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)

    def __getitem__(self, index):
        """Override."""
        s = slice(self._i0, self._i0 + self._len)
        return self._x[s][index], self._y[s][index]

    def data(self):
        """Override."""
        s = slice(self._i0, self._i0 + self._len)
        return self._x[s], self._y[s]

    def append(self, item):
        """Override."""
        x, y = item

        max_len = self._max_len
        self._x[self._i0 + self._len] = x
        self._y[self._i0 + self._len] = y
        if self._len < max_len:
            self._len += 1
        else:
            self._i0 += 1
            if self._i0 == max_len:
                self._i0 = 0
                self._x[:max_len] = self._x[max_len:]
                self._y[:max_len] = self._y[max_len:]

    def extend(self, items):
        """Override."""
        for item in items:
            self.append(item)

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)
        self._y.fill(0)

    @classmethod
    def from_array(cls, ax, ay, *args, **kwargs):
        if len(ax) != len(ay):
            raise ValueError(f"ax and ay must have the same length. "
                             f"Actual: {len(ax)}, {len(ay)}")

        instance = cls(*args, **kwargs)
        for x, y in zip(ax, ay):
            instance.append((x, y))
        return instance


_StatDataItem = namedtuple('_StatDataItem', ['avg', 'min', 'max', 'count'])


class OneWayAccuPairSequence(_AbstractSequence):
    """Store the history a pair of accumulative scalar data.

    Each data point is pair of data: (x, _StatDataItem).

    The data is collected in a stop-and-collected way. A motor, for
    example, will stop in a location and collect data for a period
    of time. Then, each data point in the accumulated pair data is
    the average of the data during this period.
    """

    def __init__(self, resolution, *,
                 max_len=3000, dtype=np.float64, min_count=2):
        super().__init__(max_len=max_len)

        self._min_count = min_count

        if resolution <= 0:
            raise ValueError("resolution must be positive!")
        self._resolution = resolution

        self._x_avg = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)
        self._count = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=np.uint64)
        self._y_avg = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y_min = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y_max = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y_std = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)

        self._last = 0

    @property
    def resolution(self):
        return self._resolution

    def __getitem__(self, index):
        """Override."""
        s = slice(self._i0, self._i0 + self._len)

        x = self._x_avg[s][index]
        y = _StatDataItem(self._y_avg[s][index],
                          self._y_min[s][index],
                          self._y_max[s][index],
                          self._count[s][index])
        return x, y

    def data(self):
        """Override."""
        last = self._i0 + self._len - 1
        if self._len > 0 and self._count[last] < self._min_count:
            s = slice(self._i0, last)
        else:
            s = slice(self._i0, last + 1)

        x = self._x_avg[s]
        y = _StatDataItem(self._y_avg[s],
                          self._y_min[s],
                          self._y_max[s],
                          self._count[s])
        return x, y

    def append(self, item):
        """Override."""
        x, y = item

        new_pt = False
        last = self._last
        if self._len > 0 or self._count[0] > 0:
            if abs(x - self._x_avg[last]) <= self._resolution:
                self._count[last] += 1
                self._x_avg[last] += (x - self._x_avg[last]) / self._count[last]
                avg_prev = self._y_avg[last]
                self._y_avg[last] += (y - self._y_avg[last]) / self._count[last]
                self._y_std[last] += (y - avg_prev)*(y - self._y_avg[last])
                # self._y_min and self._y_max does not store min and max
                # Only Standard deviation will be plotted. Min Max functionality
                # does not exist as of now.
                # self._y_min stores y_avg - 0.5*std_dev
                # self._y_max stores y_avg + 0.5*std_dev
                self._y_min[last] = self._y_avg[last] - 0.5*np.sqrt(
                    self._y_std[last]/self._count[last])
                self._y_max[last] = self._y_avg[last] + 0.5*np.sqrt(
                    self._y_std[last]/self._count[last])

                if self._count[last] == self._min_count:
                    new_pt = True

            else:
                # If the number of data at a location is less than
                # min_count, the data at this location will be discarded.
                if self._count[last] >= self._min_count:
                    self._last += 1
                    last = self._last

                self._x_avg[last] = x
                self._count[last] = 1
                self._y_avg[last] = y
                self._y_min[last] = y
                self._y_max[last] = y
                self._y_std[last] = 0.0

        else:
            self._x_avg[0] = x
            self._count[0] = 1
            self._y_avg[0] = y
            self._y_min[0] = y
            self._y_max[0] = y
            self._y_std[0] = 0.0

        if new_pt:
            max_len = self._max_len
            if self._len < max_len:
                self._len += 1
            else:
                self._i0 += 1
                if self._i0 == max_len:
                    self._i0 = 0
                    self._last -= max_len
                    self._x_avg[:max_len] = self._x_avg[max_len:]
                    self._count[:max_len] = self._count[max_len:]
                    self._y_avg[:max_len] = self._y_avg[max_len:]
                    self._y_min[:max_len] = self._y_min[max_len:]
                    self._y_max[:max_len] = self._y_max[max_len:]
                    self._y_std[:max_len] = self._y_std[max_len:]

    def append_dry(self, x):
        """Return whether append the given item will start a new position."""
        next_pos = False
        if self._len > 0 or self._count[0] > 0:
            if abs(x - self._x_avg[self._last]) > self._resolution:
                next_pos = True
        else:
            next_pos = True

        return next_pos

    def extend(self, items):
        """Override."""
        for item in items:
            self.append(item)

    def reset(self):
        """Overload."""
        self._i0 = 0
        self._len = 0
        self._last = 0
        self._x_avg.fill(0)
        self._count.fill(0)
        self._y_avg.fill(0)
        self._y_min.fill(0)
        self._y_max.fill(0)
        self._y_std.fill(0)

    @classmethod
    def from_array(cls, ax, ay, *args, **kwargs):
        if len(ax) != len(ay):
            raise ValueError(f"ax and ay must have the same length. "
                             f"Actual: {len(ax)}, {len(ay)}")

        instance = cls(*args, **kwargs)
        for x, y in zip(ax, ay):
            instance.append((x, y))
        return instance
