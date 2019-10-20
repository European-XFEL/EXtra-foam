"""
Offline and online data analysis and visualization tool for azimuthal
integration of different data acquired with various detectors at
European XFEL.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from collections import OrderedDict
from itertools import chain

from karaboFAI.config import config


_MISC_CATEGORIES = [
    "XGM",
    "Digitizer",
    "Monochromator",
    "Motor",
    "Magnet"
]

# The source names of image detectors will be injected at run time.

# the "set"s will be converted to list at last
DATA_SOURCE_CATEGORIES = {
    "UNKNOWN": {
        key: set() for key in chain(sorted(config.detectors), _MISC_CATEGORIES)
    },
    "SPB": {
        "AGIPD": set(),
    },
    "FXE": {
        "LPD": {
            "FXE_DET_LPD1M-1/DET/*CH0:xtdf"
        },
        "JungFrau": set(),
        "JungFrauPR": set(),
        "XGM": {
            "SA1_XTD2_XGM/DOOCS/MAIN",
            "SA1_XTD2_XGM/DOOCS/MAIN:output",
            "SPB_XTD9_XGM/DOOCS/MAIN",
            "SPB_XTD9_XGM/DOOCS/MAIN:output",
        },
        "Motor": {
            "FXE_SMS_USR/MOTOR/UM01",
            "FXE_SMS_USR/MOTOR/UM02",
            "FXE_SMS_USR/MOTOR/UM04",
            "FXE_SMS_USR/MOTOR/UM05",
            "FXE_SMS_USR/MOTOR/UM13",
            "FXE_AUXT_LIC/DOOCS/PPLASER",
            "FXE_AUXT_LIC/DOOCS/PPODL",
        },
    },
    "SCS": {
        "DSSC": {
            "SCS_DET_DSSC1M-1/DET/*CH0:xtdf",
        },
        "FastCCD": set(),
        "XGM": {
            "SA3_XTD10_XGM/XGM/DOOCS",
            "SA3_XTD10_XGM/XGM/DOOCS:output",
            "SCS_BLU_XGM/XGM/DOOCS",
            "SCS_BLU_XGM/XGM/DOOCS:output"
        },
        "Magnet": {
            "SCS_CDIFFT_MAG/SUPPLY/CURRENT",
        },
        "MonoChromator": {
            "SA3_XTD10_MONO/MDL/PHOTON_ENERGY",
        },
        "Motor": {
            "SCS_ILH_LAS/DOOCS/PP800_PHASESHIFTER",
            "SCS_ILH_LAS/MOTOR/LT3",
        },
    },
    "SQS": {

    },
    "MID": {
        "AGIPD": set()
    },
    "HED": {
        "JungFrau": set(),
    },
}

EXCLUSIVE_SOURCE_CATEGORIES = config.detectors.copy()

DATA_SOURCE_PROPERTIES = {
    "AGIPD": [
        "image.data"
    ],
    "AGIPD:xtdf": [
        "image.data"
    ],
    "LPD": [
        "image.data"
    ],
    "LPD:xtdf": [
        "image.data"
    ],
    "DSSC": [
        "image.data"
    ],
    "DSSC:xtdf": {
        "image.data"
    },
    "JungFrau": [
        "image.data"
    ],
    "JungFrau:display": [
        "image.data"
    ],
    "JungFrau:daqOutput": [
        "image.data"
    ],
    "JungFrauPR": [
        "image.data"
    ],
    "JungFrauPR:display": [
        "image.data"
    ],
    "JungFrauPR:daqOutput": [
        "image.data"
    ],
    "FastCCD:output": [
        "image.data"
    ],
    "FastCCD:daqOutput": [
        "data.image.pixels"
    ],
    "XGM": [
        "photonEnergy.photonFlux",
    ],
    "XGM:output": [
        "intensityTD",
        "sa1IntensityTD",
        "sa3IntensityTD",
    ],
    "Magnet": [
        "actualCurrent",
    ],
    "MonoChromator": [
        "actualEnergy",
    ],
    "Motor": [
        "actualPosition",
    ],
    "Metadata": [
        "timestamp.tid",
    ]
}


def _sort_dict(data):
    """Sort a dictionary by key.

    :param dict data: dictionary.
    :return OrderedDict data: sorted dictionary.
    """
    sorted_dict = OrderedDict()
    for key in sorted(data):
        sorted_dict[key] = data[key]

    return sorted_dict


# add common categories
for topic in DATA_SOURCE_CATEGORIES:
    topic_data = DATA_SOURCE_CATEGORIES[topic]

    for ctg in _MISC_CATEGORIES:
        if ctg not in DATA_SOURCE_CATEGORIES[topic]:
            topic_data[ctg] = []
        else:
            topic_data[ctg] = sorted(topic_data[ctg])

    topic_data["Metadata"] = {
        "Metadata",
    }

    DATA_SOURCE_CATEGORIES[topic] = _sort_dict(DATA_SOURCE_CATEGORIES[topic])


class SourceItem:
    def __init__(self, category, name, ppt):
        self.category = category
        self.name = name
        self.property = ppt
