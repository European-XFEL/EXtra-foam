"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import os
import os.path as osp
import shutil

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGridLayout

from .base_view import _AbstractImageToolView, create_imagetool_view
from ..ctrl_widgets import RefImageCtrlWidget
from ..plot_widgets import ImageViewF
from ...algorithms import movingAvgImageData
from ...file_io import write_image
from ...ipc import ReferencePub
from ...logger import logger
from ... import ROOT_PATH

REFERENCE_FILE = osp.join(ROOT_PATH, "tmp", ".reference.npy")


@create_imagetool_view(RefImageCtrlWidget)
class ReferenceView(_AbstractImageToolView):
    """ReferenceView class.

    Widget for visualizing the reference image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._corrected = ImageViewF(hide_axis=False)
        self._corrected.setTitle("Current")
        self._reference = ImageViewF(hide_axis=False)
        self._reference.setTitle("Reference")

        self._pub = ReferencePub()

        self._is_recording = False
        self._recorded_image = None
        self._count = 0

        self.initUI()
        self.initConnections()

    def initUI(self):
        """Override."""
        layout = QGridLayout()
        layout.addWidget(self._corrected, 0, 0)
        layout.addWidget(self._reference, 0, 1)
        layout.addWidget(self._ctrl_widget, 1, 0, 1, 2)
        layout.setContentsMargins(1, 1, 1, 1)
        self.setLayout(layout)

    def initConnections(self):
        """Override."""
        self._ctrl_widget.load_btn.clicked.connect(self._loadReference)
        self._ctrl_widget.set_current_btn.clicked.connect(
            self._setCurrentAsReference)
        self._ctrl_widget.remove_btn.clicked.connect(self._removeReference)
        self._ctrl_widget.record_btn.toggled.connect(self._recordReference)
        self._ctrl_widget.save_btn.clicked.connect(self._saveReference)

    def updateF(self, data, auto_update):
        """Override."""
        if auto_update or self._corrected.image is None:
            self._corrected.setImage(data.image.masked_mean)
            # Removing and displaying of the currently displayed image
            # is deferred.
            self._reference.setImage(data.image.reference)

        if self._is_recording:
            self._count += 1
            # Update cached image and count
            image = self._corrected.image
            if self._recorded_image is None:
                self._recorded_image = image.copy()
            else:
                # Average without storing images
                movingAvgImageData(self._recorded_image, image, self._count)

            # Update status display
            status = f"Recording..\t\tUsed trains: {self._count}"
            self._ctrl_widget.record_label.setText(status)

    @pyqtSlot()
    def _loadReference(self):
        """Load the reference image from a file."""
        filepath = QFileDialog.getOpenFileName(
            caption="Load reference image",
            directory=osp.expanduser("~"))[0]

        # do not remove reference if the user meant to cancel the selection
        if filepath:
            self._pub.set(filepath)
            self._ctrl_widget.filepath_le.setText(filepath)
            self._ctrl_widget.save_btn.setDisabled(True)
            # Update label
            text = "The loaded file has been applied as reference."
            self._ctrl_widget.record_label.setText(text)

    @pyqtSlot()
    def _setCurrentAsReference(self):
        """Set the current corrected image as reference."""
        image = self._corrected.image
        if image is not None:
            self._writeTempReferenceFile(image)
            if self._checkReferenceFile():
                self._setReference(REFERENCE_FILE)
                # Update label
                text = "The current image has been applied as reference."
                self._ctrl_widget.record_label.setText(text)

    def _writeTempReferenceFile(self, image):
        """Save the input image as a reference file (.npy).
           Path is `REFERENCE_FILE`"""
        if image is None:
            return

        os.makedirs(osp.dirname(REFERENCE_FILE), exist_ok=True)

        try:
            write_image(REFERENCE_FILE, image)
        except ValueError as e:
            logger.error(str(e))
            # Remove previous reference file
            if osp.exists(REFERENCE_FILE):
                os.remove(REFERENCE_FILE)

    def _setReference(self, path):
        """Sets the input image as reference"""
        self._pub.set(path)
        self._ctrl_widget.filepath_le.setText(path)

    @pyqtSlot()
    def _removeReference(self):
        """Remove the reference image."""
        self._setReference("")
        self._ctrl_widget.record_label.setText("")

    @pyqtSlot(bool)
    def _recordReference(self, is_recording):
        """Start/stop the recording of the reference file.

           When starting, a flag will be set for the `updateF()` to detect the
           start of the image averaging of the received data.
           Upon finishing, the recorded image is written as reference file.
        """
        self._is_recording = is_recording

        # Log recording status
        status = "started" if is_recording else "finished"
        logger.info(f"Recording reference has {status}.")

        if is_recording:
            # Remove existing reference file
            self._removeReference()
        elif self._recorded_image is not None:
            # Save recording to a temp file upon finishing
            self._writeTempReferenceFile(self._recorded_image)
            if self._checkReferenceFile():
                self._setReference(REFERENCE_FILE)

            # Reset widget and variables
            self._recorded_image = None
            self._count = 0

            # Update label
            text = ("The average of recorded trains "
                    "has been applied as reference.")
            self._ctrl_widget.record_label.setText(text)

    @pyqtSlot()
    def _saveReference(self):
        """'Save' the reference file on the user supplied path by conveniently
            copying the existing file to the desired path."""
        if not self._checkReferenceFile():
            return

        filepath = QFileDialog.getSaveFileName(
            caption="Save image",
            directory=osp.expanduser("~"),
            filter="NumPy Binary File (*.npy)")[0]

        # Validate filepath
        if not filepath:
            return
        if not filepath.lower().endswith(".npy"):
            filepath += ".npy"

        # Copy reference file from tmp folder to desired destination
        os.makedirs(osp.dirname(filepath), exist_ok=True)
        shutil.copyfile(REFERENCE_FILE, filepath)

    def _checkReferenceFile(self, path=REFERENCE_FILE):
        """Check if the reference file exists."""
        exists = osp.exists(path)
        self._ctrl_widget.save_btn.setDisabled(not exists)
        return exists
