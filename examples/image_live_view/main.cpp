/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */
#include <QApplication>

#include "image_view.hpp"


int main(int argc, char* argv[])
{
  QApplication app(argc, argv);

  ImageView view;
  view.show();

  return app.exec();
}
