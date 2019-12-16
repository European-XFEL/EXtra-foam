"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Sequence

import numpy as np

from ..exceptions import ProcessingError
from ...database import MetaProxy
from ...algorithms import normalize_auc
from ...config import AnalysisType, Normalizer


class State(ABC):
    """Base class of processor state."""
    @abstractmethod
    def on_enter(self, proc):
        pass

    @abstractmethod
    def next(self):
        """Return the next state."""
        pass

    def update(self, proc):
        self.on_enter(proc)


class StateOn(State):
    """State on.

    The state when the processor is ready to start.
    """
    def on_enter(self, proc):
        handler = proc.on_handler
        if handler is not None:
            handler(proc)

    def next(self):
        return StateProcessing()


class StateProcessing(State):
    """State processing.

    The state when the processor is processing data.
    """
    def on_enter(self, proc):
        handler = proc.processing_handler
        if handler is not None:
            handler(proc)

    def next(self):
        return StateOn()


class SharedProperty:
    """Property shared among Processors.

    Define a property which is shared by the current processor and its
    children processors.
    """
    def __init__(self):
        self.name = None

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        if self.name not in instance._params:
            # initialization
            instance._params[self.name] = None

        return instance._params[self.name]

    def __set__(self, instance, value):
        instance._params[self.name] = value


class _RedisParserMixin:
    """_RedisParserMixin class.

    Due to the performance concern, methods in this class are not suppose
    to cover all the corner cases, passing an arbitrary input may result
    in undefined behavior.
    """
    @staticmethod
    def str2tuple(text, delimiter=",", handler=float):
        """Convert a string to a tuple.

        The string is expected to be the result of str(tp), where tp is a
        tuple.

        For example:
            str2tuple('(1, 2)') -> (1.0, 2.0)
        """
        splitted = text[1:-1].split(delimiter)
        return handler(splitted[0]), handler(splitted[1])

    @staticmethod
    def str2list(text, delimiter=",", handler=float):
        """Convert a string to a list.

        The string is expected to be the result of str(lt), where lt is a
        list.

        For example:
            str2list('[1, 2, 3]') -> [1.0, 2.0, 3.0]
        """
        if not text[1:-1]:
            return []
        return [handler(v) for v in text[1:-1].split(delimiter)]

    @staticmethod
    def str2slice(text):
        """Convert a string to a slice object.

        The string is expected to the result of str(lt), where lt can be
        converted to a slice object by slice(*lt).

        For example:
            str2slice('[None, 2]' -> slice(None, 2)
        """
        return slice(*[None if v.strip() == 'None' else int(v)
                       for v in text[1:-1].split(',')])


class MetaProcessor(type):
    def __new__(mcs, name, bases, class_dict):
        for key, value in class_dict.items():
            if isinstance(value, SharedProperty):
                value.name = key

        cls = type.__new__(mcs, name, bases, class_dict)
        return cls


class _BaseProcessor(_RedisParserMixin, metaclass=MetaProcessor):
    """Data processor interface."""

    def __init__(self):
        self._meta = MetaProxy()

    def _update_analysis(self, analysis_type, *, register=True):
        """Update analysis type.

        :param AnalysisType analysis_type: analysis type.
        :param bool register: True for (un)register the analysis type.

        :return: True if the analysis type has changed and False for not.
        """
        if not isinstance(analysis_type, AnalysisType):
            raise ProcessingError(
                f"Unknown analysis type: {str(analysis_type)}")

        if analysis_type != self.analysis_type:
            if register:
                # unregister the old
                if self.analysis_type is not None:
                    self._meta.unregister_analysis(self.analysis_type)

                # register the new one
                if analysis_type != AnalysisType.UNDEFINED:
                    self._meta.register_analysis(analysis_type)

            self.analysis_type = analysis_type
            return True

        return False

    def run_once(self, data):
        """Composition interface.

        :param dict data: data which contains raw and processed data, etc.
        """
        self.update()
        self.process(data)

    def update(self):
        """Update metadata."""
        raise NotImplementedError

    def process(self, data):
        """Process data.

        :param dict data: data which contains raw and processed data, etc.
        """
        raise NotImplementedError

    @staticmethod
    def _normalize_fom(processed, y, normalizer, *, x=None, auc_range=None):
        """Normalize FOM/VFOM.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray y: y values.
        :param Normalizer normalizer: normalizer type.
        :param numpy.ndarray x: x values used with AUC normalizer..
        :param tuple auc_range: normalization range with AUC normalizer.
        """
        if normalizer == Normalizer.UNDEFINED:
            return y

        if normalizer == Normalizer.AUC:
            # normalized by area under curve (AUC)
            normalized = normalize_auc(y, x, auc_range)
        elif normalizer == Normalizer.XGM:
            # normalized by XGM
            intensity = processed.pulse.xgm.intensity
            if intensity is None:
                raise ProcessingError("XGM normalizer is not available!")
            denominator = np.mean(intensity)

            if denominator == 0:
                raise ProcessingError("XGM normalizer is zero!")

            normalized = y / denominator
        elif normalizer == Normalizer.ROI:
            # normalized by ROI
            denominator = processed.roi.norm

            if denominator is None:
                raise ProcessingError("ROI normalizer is not available!")

            if denominator == 0:
                raise ProcessingError("ROI normalizer is zero!")

            normalized = y / denominator

        else:
            raise ProcessingError(f"Unknown normalizer: {repr(normalizer)}")

        return normalized

    @staticmethod
    def _normalize_fom_pp(processed, y_on, y_off, normalizer, *,
                          x=None, auc_range=None):
        """Normalize pump-probe FOM/VFOM.

        :param ProcessedData processed: processed data.
        :param numpy.ndarray y_on: pump y values.
        :param numpy.ndarray y_off: probe y values.
        :param Normalizer normalizer: normalizer type.
        :param numpy.ndarray x: x values used with AUC normalizer..
        :param tuple auc_range: normalization range with AUC normalizer.
        """
        if normalizer == Normalizer.UNDEFINED:
            return y_on, y_off

        if normalizer == Normalizer.AUC:
            # normalized by area under curve (AUC)
            normalized_on = normalize_auc(y_on, x, auc_range)
            normalized_off = normalize_auc(y_off, x, auc_range)
        elif normalizer == Normalizer.XGM:
            # normalized by XGM
            denominator_on = processed.xgm.on.intensity
            denominator_off = processed.xgm.off.intensity

            if denominator_on is None or denominator_off is None:
                raise ProcessingError("XGM normalizer is not available!")

            if denominator_on == 0:
                raise ProcessingError("XGM normalizer (on) is zero!")

            if denominator_off == 0:
                raise ProcessingError("XGM normalizer (off) is zero!")

            normalized_on = y_on / denominator_on
            normalized_off = y_off / denominator_off

        elif normalizer == Normalizer.ROI:
            # normalized by ROI
            denominator_on = processed.pp.roi_norm_on
            denominator_off = processed.pp.roi_norm_off

            if denominator_on is None:
                raise ProcessingError("ROI normalizer (on) is not available!")

            if denominator_off is None:
                raise ProcessingError("ROI normalizer (off) is not available!")

            if denominator_on == 0:
                raise ProcessingError("ROI normalizer (on) is zero!")

            if denominator_off == 0:
                raise ProcessingError("ROI normalizer (off) is zero!")

            normalized_on = y_on / denominator_on
            normalized_off = y_off / denominator_off

        else:
            raise ProcessingError(f"Unknown normalizer: {repr(normalizer)}")

        return normalized_on, normalized_off

    @staticmethod
    def _fetch_property_data(tid, raw, src_name, ppt):
        """Fetch property data from raw data.

        :param int tid: train ID.
        :param dict raw: raw data.
        :param str src_name: device ID.
        :param str ppt: property name.

        :returns (value, error str)
        """
        if not src_name or not ppt:
            # not activated is not an error
            return None, ""

        if src_name == "Any":
            return tid, ""
        else:
            try:
                device_data = raw[src_name]
            except KeyError:
                return None, f"[{tid}] source '{src_name}' is not in the data!"

            ppt_orig = ppt
            try:
                if ppt not in device_data:
                    # instrument data from file
                    ppt += '.value'
                return device_data[ppt], ""
            except KeyError:
                return None, f"[{tid}] '{src_name}' does not contain " \
                             f"property '{ppt_orig}'"


class _AbstractSequence(Sequence):
    """Base class for 'Sequence' data."""
    _OVER_CAPACITY = 2

    def __init__(self, max_len=3000):
        self._max_len = max_len

        self._i0 = 0  # starting index
        self._len = 0

    def __len__(self):
        """Override."""
        return self._len

    @abstractmethod
    def data(self):
        """Return all the data."""
        pass

    @abstractmethod
    def append(self, item):
        """Add a new data point."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the data history."""
        pass


class SimpleSequence(_AbstractSequence):
    """Store the history of scalar data."""

    def __init__(self, *, max_len=100000, dtype=np.float64):
        super().__init__(max_len=max_len)

        self._x = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)

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

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)

    def extend(self, v_lst):
        # TODO: improve
        for v in v_lst:
            self.append(v)


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

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)


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

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)
        self._y.fill(0)


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
                 max_len=3000, dtype=np.float64, min_count=2, epsilon=1.e-9):
        super().__init__(max_len=max_len)

        self._min_count = min_count
        self._epsilon = np.abs(epsilon)

        if resolution <= 0:
            raise ValueError("'resolution must be positive!")
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
        if self._len > 0:
            last = self._i0 + self._len - 1
            if abs(x - self._x_avg[last]) - self._resolution < self._epsilon:
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

            else:
                # If the number of data at a location is less than
                # min_count, the data at this location will be discarded.
                if self._count[last] >= self._min_count:
                    new_pt = True
                    last += 1

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
            new_pt = True

        if new_pt:
            max_len = self._max_len
            if self._len < max_len:
                self._len += 1
            else:
                self._i0 += 1
                if self._i0 == max_len:
                    self._x_avg[:max_len] = self._x_avg[max_len:]
                    self._count[:max_len] = self._count[max_len:]
                    self._y_avg[:max_len] = self._y_avg[max_len:]
                    self._y_min[:max_len] = self._y_min[max_len:]
                    self._y_max[:max_len] = self._y_max[max_len:]
                    self._y_std[:max_len] = self._y_std[max_len:]

    def reset(self):
        """Overload."""
        self._i0 = 0
        self._len = 0
        self._x_avg.fill(0)
        self._count.fill(0)
        self._y_avg.fill(0)
        self._y_min.fill(0)
        self._y_max.fill(0)
        self._y_std.fill(0)
