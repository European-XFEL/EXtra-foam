"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import argparse
import faulthandler

from . import logger, mkQApp, __version__
from .config import config
from .facade import SpecialSuiteController


def application():
    parser = argparse.ArgumentParser(prog="extra-foam-special-suite")
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s " + __version__)
    parser.add_argument("topic", help="Name of the topic",
                        choices=config.topics,
                        type=lambda s: s.upper())
    parser.add_argument("--use-gate", action='store_true',
                        help="Use Karabo gate client (experimental feature)")
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel("DEBUG")
    # No ideal whether it affects the performance. If it does, enable it only
    # in debug mode.
    faulthandler.enable(all_threads=False)

    topic = args.topic

    config.load(topic, USE_KARABO_GATE_CLIENT=args.use_gate)

    app = mkQApp()
    app.setStyleSheet(
        "QTabWidget::pane { border: 0; }"
    )

    controller = SpecialSuiteController()
    controller.showFacade(topic)

    app.exec_()


if __name__ == "__main__":

    application()
