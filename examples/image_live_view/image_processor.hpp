/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef LIVE_VIEW_IMAGE_PROCESSOR_H
#define LIVE_VIEW_IMAGE_PROCESSOR_H

#include <deque>
#include <memory>

#include <QObject>

#include "config.hpp"


class ImageProcessor : public QObject
{
  Q_OBJECT

  std::shared_ptr<ImageQueue> queue_;

  bool processing_;

public:
  explicit ImageProcessor(const std::shared_ptr<ImageQueue>& queue, QObject* parent = nullptr);

  ~ImageProcessor() override = default;

  void stop();

signals:
  // emitted when a new frame is ready
  void newFrame(QPixmap pix);

public slots:
  void process();
};


#endif //LIVE_VIEW_IMAGE_PROCESSOR_H
