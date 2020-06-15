/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef LIVE_VIEW_IMAGEVIEW_H
#define LIVE_VIEW_IMAGEVIEW_H

#include <QMainWindow>
#include <QAction>
#include <QLabel>
#include <QToolBar>
#include <QThread>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>

#include "broker.hpp"
#include "image_processor.hpp"
#include "config.hpp"


class ImageView : public QMainWindow
{
  Q_OBJECT

  static constexpr size_t width_ = 1280;
  static constexpr size_t height_ = 800;

  QGraphicsPixmapItem displayed_;
  QGraphicsView* view_;
  QGraphicsScene* scene_;

  Broker* broker_;
  QThread* broker_t_;

  ImageProcessor* image_proc_;
  QThread* image_proc_t_;

  QToolBar* tool_bar_;

  QAction* start_act_;
  QAction* stop_act_;

  std::shared_ptr<ImageQueue> queue_;

public:
  explicit ImageView(QWidget *parent = nullptr);

  ~ImageView() override;

  void updateImage(QPixmap pix);

private:
  void initUI();
  void initConnections();

private slots:
  void onStart();
  void onStop();
};


#endif //LIVE_VIEW_IMAGEVIEW_H
