/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef LIVE_VIEW_BRIDGE_H
#define LIVE_VIEW_BRIDGE_H

#include <memory>
#include <deque>

#include <QThread>
#include <QString>
#include <QPixmap>
#include <QSet>
#include <QMutex>

#include <karabo-bridge/kb_client.hpp>

#include "config.hpp"


class Broker : public QObject
{
  Q_OBJECT

  bool acquiring_;

  std::string endpoint_; // TCP address of the endpoint
  std::string output_ch_; // output channel name
  std::string ppt_; // property name

  std::shared_ptr<ImageQueue> queue_;

public:
  explicit Broker(const std::shared_ptr<ImageQueue>& queue, QObject* parent = nullptr);

  ~Broker() override = default;

  void recv();

  void stop();
};


#endif //LIVE_VIEW_BRIDGE_H
