/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include <array>
#include <algorithm>

#include <QDebug>
#include <QMutexLocker>

#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>

#include "broker.hpp"


template<typename S>
std::array<size_t, 2> checkImageShape(S&& shape)
{
  if (shape.size() == 2) return {shape[0], shape[1]};
  if (shape.size() == 3)
  {
    if (shape[0] == 1) return {shape[1], shape[2]};
    if (shape[2] == 1) return {shape[0], shape[1]};
  }

  return {0, 0};
}


Broker::Broker(const std::shared_ptr<ImageQueue>& queue, QObject *parent)
  : queue_(queue), acquiring_(false)
{
  Q_UNUSED(parent)

  endpoint_ = "tcp://127.0.0.1:45453";
  output_ch_ = "FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput";
  ppt_ = "data.adc";
}

void Broker::recv()
{
  acquiring_ = true;

  karabo_bridge::Client client(0.1); // 100 ms timeout
  try
  {
    client.connect(endpoint_);
    qDebug() << "Connected to server: " << endpoint_.c_str();
  } catch(const std::exception& e)
  {
    qDebug() << "Failed to connect to server: " << e.what();
    return;
  }

  while (acquiring_)
  {
    qDebug() << "Request data from bridge ...";
    std::map<std::string, karabo_bridge::kb_data> data_pkg = client.next();

    if (data_pkg.empty())
    {
      std::cerr << "No data received!\n";
    } else
    {
      // find data for the output channel
      auto it = data_pkg.find(output_ch_);
      if (it != data_pkg.end())
      {
        // get property data
        auto arr = it->second.array;
        auto it2 = arr.find(ppt_);
        if (it2 != arr.end())
        {
          auto data = it2->second;

          // check whether the data is a 2D array or can be squeezed into a 2D array
          auto shape = checkImageShape(data.shape());
          if (shape[0] > 0)
          {
            auto tid = it->second.metadata["timestamp.tid"].as<uint64_t>();

            std::cout << "Train " << tid
                      << ": Found image data with shape (" << shape[0] << ", " << shape[1] << ")\n";

            // data is copied into the queue for further processing
            queue_->push(xt::adapt(static_cast<float*>(data.data()), shape));
            continue;
          } else
          {
            std::stringstream buffer;
            std::copy(shape.begin(), shape.end(), std::ostream_iterator<size_t>(buffer, ", "));
            std::cerr << "Not supported data shape: (" + buffer.str() + ")\n";
          }

        } else
        {
          std::cerr << "Not found: data with property '" << ppt_ << "'\n";
        }
      } else
      {
        std::cerr << "Not found: data with output channel '" << output_ch_ << "'\n";
      }
    }

    QThread::msleep(1);
  }
}

void Broker::stop()
{
  acquiring_ = false;
}
