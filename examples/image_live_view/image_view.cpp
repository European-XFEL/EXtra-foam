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
#include <QStatusBar>
#include <QLayout>
#include <QDebug>

#include "image_view.hpp"
#include "config.hpp"


ImageView::ImageView(QWidget *parent)
  : QMainWindow(parent), queue_(std::make_shared<ImageQueue>())
{
  initUI();
  initConnections();
}

ImageView::~ImageView()
{
  onStop();

  image_proc_t_->wait();
  broker_t_->wait();
}

void ImageView::initUI()
{
  // tool bar
  tool_bar_ = addToolBar("View");

  start_act_ = new QAction("&Start", this);
  stop_act_ = new QAction("&Stop", this);
  stop_act_->setEnabled(false);

  tool_bar_->addAction(start_act_);
  tool_bar_->addAction(stop_act_);

  // view
  scene_ = new QGraphicsScene();
  view_ = new QGraphicsView(scene_);
  view_->scene()->addItem(&displayed_);
  setCentralWidget(view_);

  // others
  this->setWindowTitle("EXtra-foam C++ API example: Image view");
  this->resize(width_, height_);
  setMinimumSize(640, 480);
}

void ImageView::initConnections()
{
  connect(start_act_, &QAction::triggered, this, &ImageView::onStart);
  connect(stop_act_, &QAction::triggered, this, &ImageView::onStop);

  broker_ = new Broker(queue_);
  broker_t_ = new QThread;
  connect(broker_t_, &QThread::started, broker_, &Broker::recv);
  broker_->moveToThread(broker_t_);

  image_proc_ = new ImageProcessor(queue_);
  connect(image_proc_, &ImageProcessor::newFrame, this, &ImageView::updateImage);
  image_proc_t_ = new QThread;
  connect(image_proc_t_, &QThread::started, image_proc_, &ImageProcessor::process);
  image_proc_->moveToThread(image_proc_t_);
}

void ImageView::onStart()
{
  start_act_->setEnabled(false);
  stop_act_->setEnabled(true);

  broker_t_->start();
  image_proc_t_->start();
}

void ImageView::onStop()
{
  stop_act_->setEnabled(false);
  start_act_->setEnabled(true);

  broker_->stop();
  image_proc_->stop();

  broker_t_->quit();
  image_proc_t_->quit();
}

void ImageView::updateImage(QPixmap pix)
{
  displayed_.setPixmap(pix);
  qDebug() << "Image updated";
}