"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Cammille Carinan <cammille.carinan@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from extra_foam.algorithms import hist_with_stats


@dataclass
class Device:
    name: str
    value: float


@dataclass
class Projection:
    x: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    fit: np.ndarray = np.array([])
    pos: float = 0.0
    width: float = 0.0


@dataclass
class Histogram:
    count: np.array
    bins: np.array
    mean: float
    median: float
    std: float


@dataclass
class Measurement:
    mean: float
    std: float


@dataclass
class TrainMeasurement:
    _mean: List[float] = field(default_factory=list)
    _std: List[float] = field(default_factory=list)

    _mean_hist: Optional[Histogram] = field(init=False)
    _std_hist: Optional[Histogram] = field(init=False)

    def update(self, values: Measurement):
        self._mean.append(values.mean)
        self._std.append(values.std)

    def calc_histogram(self, n_bins=10):
        self._mean_hist = self._histogram(self.mean, n_bins=n_bins)
        self._std_hist = self._histogram(self.std, n_bins=n_bins)

    def reset(self):
        self._mean = []
        self._std = []

    @property
    def mean(self):
        return np.array(self._mean)

    @property
    def std(self):
        return np.array(self._std)

    @property
    def data(self):
        return {
            "mean": self.mean,
            "mean_hist": self._mean_hist,
            "std": self.std,
            "std_hist": self._std_hist
        }

    @staticmethod
    def _histogram(data, n_bins=10):
        data = data[np.isfinite(data)]
        return Histogram(*hist_with_stats(data, n_bins=n_bins))


@dataclass
class PulseMeasurement:
    values: List[List[float]] = field(default_factory=list)

    def update(self, values):
        self.values.append(values)

    def reset(self):
        self.values = []

    def get_values(self, pulse_num: int = 0):
        return np.array([value[pulse_num] for value in self.values])

    @property
    def data(self):
        return {
            "values": np.array(self.values),
            "mean": np.mean(self.values, axis=0),
            "std": np.std(self.values, axis=0)
        }


@dataclass
class ViewData:
    image: np.ndarray = np.array([])
    proj: Projection = field(default_factory=Projection)
    widths: np.ndarray = np.array([])

    @property
    def data(self):
        return {
            "image": self.image,
            "proj": self.proj,
            "widths": self.widths
        }


@dataclass
class Pulse:
    """Pulse data"""
    pos: float
    width: float


@dataclass
class Train:
    """Train data"""
    # Metadata
    trainId: int
    device: Optional[Device]

    # Processed data
    pulses: List[Pulse]

    pos_mean: float = field(default=None)
    pos_std: float = field(default=None)
    width_mean: float = field(default=None)
    width_std: float = field(default=None)

    @property
    def pos(self):
        return self._measurement([pulse.pos for pulse in self.pulses])

    @property
    def width(self):
        return self._measurement([pulse.width for pulse in self.pulses])

    def _measurement(self, measure: Sequence):
        return Measurement(mean=np.mean(measure), std=np.std(measure))


@dataclass
class Trains:
    trainIds: List[int] = field(default_factory=list)
    initial_trainId: Optional[int] = field(init=False)

    device: List[float] = field(default_factory=list)

    # Metadata
    num_pulses: int = 16
    hist_bins: int = 10

    # Train-resolved measurements
    pos_train: TrainMeasurement = field(default_factory=TrainMeasurement)
    width_train: TrainMeasurement = field(default_factory=TrainMeasurement)

    # Pulse-resolved measurements
    pos_pulse: PulseMeasurement = field(default_factory=PulseMeasurement)
    width_pulse: PulseMeasurement = field(default_factory=PulseMeasurement)

    def add(self, train: Train):
        # Update trainId
        trainId = train.trainId
        if self.initial_trainId is None:
            self.initial_trainId = trainId
        self.trainIds.append(trainId - self.initial_trainId)

        # Update device value
        self.device.append(train.device.value)

        # Update train-resolve pos and width
        self.pos_train.update(train.pos)
        self.pos_train.calc_histogram(n_bins=self.hist_bins)
        self.width_train.update(train.width)

        # Update pulse-resolve pos and width
        self.pos_pulse.update([pulse.pos for pulse in train.pulses])
        self.width_pulse.update([pulse.width for pulse in train.pulses])

    def reset(self):
        self.trainIds = []
        self.initial_trainId = None
        self.device = []

        self.pos_train.reset()
        self.width_train.reset()
        self.pos_pulse.reset()
        self.width_pulse.reset()

    @property
    def data(self):
        return {
            "trainIds": np.array(self.trainIds),
            "initial_trainId": self.initial_trainId,
            "device": np.array(self.device),

            # Train-resolved data
            "pos_train": self.pos_train.data,

            # Pulse-resolved data
            "pulses": np.arange(self.num_pulses),
            "pos_pulse": self.pos_pulse.data,
            "width_pulse": self.width_pulse.data,
        }
