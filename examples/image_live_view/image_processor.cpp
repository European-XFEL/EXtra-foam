/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include <QThread>
#include <QImage>
#include <QPixmap>
#include <QDebug>

#include <xtensor/xarray.hpp>

#include "image_processor.hpp"


ImageProcessor::ImageProcessor(const std::shared_ptr<ImageQueue>& queue, QObject* parent)
  : queue_(queue), processing_(false)
{
  Q_UNUSED(parent)
}

void ImageProcessor::process()
{
  processing_ = true;
  while(processing_)
  {
    xt::xarray<float> arr;
    if (queue_->try_pop(arr))
    {
      auto data = reinterpret_cast<unsigned char*>(arr.data());
      QImage img(data, arr.shape()[1], arr.shape()[0], QImage::Format_ARGB32);
      qDebug() << "Image processed";

      emit newFrame(QPixmap::fromImage(img));
    }
    QThread::msleep(1);
  }
}

void ImageProcessor::stop()
{
  processing_ = false;
}
