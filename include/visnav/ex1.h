/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement

  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat.setZero();
  w_hat(0, 1) = -xi(2);
  w_hat(1, 0) = xi(2);
  w_hat(0, 2) = xi(1);
  w_hat(2, 0) = -xi(1);
  w_hat(1, 2) = -xi(0);
  w_hat(2, 1) = xi(0);
  double theta = xi.norm();
  Eigen::Matrix<T, 3, 3> I;
  I.setIdentity();
  Eigen::Matrix<T, 3, 3> expS =
      I + ((w_hat / theta) * sin(theta)) +
      (((w_hat * w_hat) / (theta * theta)) * (1 - cos(theta)));
  if (theta == 0) {
    expS.setIdentity();
  }

  // UNUSED(xi);
  return expS;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  double theta = acos((mat.trace() - 1) / 2);
  double k = (theta / (2 * sin(theta)));
  Eigen::Matrix<T, 3, 1> log;
  log(0, 0) = (mat(2, 1) - mat(1, 2)) * k;
  log(1, 0) = (mat(0, 2) - mat(2, 0)) * k;
  log(2, 0) = (mat(1, 0) - mat(0, 1)) * k;
  if (theta == 0) {
    log(0, 0) = log(1, 0) = log(2, 0) = 0;
  }
  // UNUSED(mat);
  return log;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 1> w;
  Eigen::Matrix<T, 3, 1> v;
  for (int i = 1; i < 4; i++) {
    v(i - 1, 0) = xi(i - 1, 0);
  }
  for (int i = 1; i < 4; i++) {
    w(i - 1, 0) = xi(i + 2, 0);
  }
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat.setZero();
  w_hat(0, 1) = -w(2);
  w_hat(1, 0) = w(2);
  w_hat(0, 2) = w(1);
  w_hat(2, 0) = -w(1);
  w_hat(1, 2) = -w(0);
  w_hat(2, 1) = w(0);
  double theta = w.norm();
  Eigen::Matrix<T, 3, 3> I;
  I.setIdentity();
  Eigen::Matrix<T, 3, 3> exp =
      I + ((w_hat / theta) * sin(theta)) +
      (((w_hat * w_hat) / (theta * theta)) * (1 - cos(theta)));
  Eigen::Matrix<T, 3, 3> J =
      Eigen::Matrix3d::Identity() + w_hat / (theta * theta) * (1 - cos(theta)) +
      ((w_hat * w_hat) / (theta * theta * theta)) * (theta - sin(theta));
  if (w.isZero()) {
    exp.setIdentity();
    J.setIdentity();
  }
  Eigen::Matrix<T, 3, 1> jv = J * v;
  Eigen::Matrix<T, 4, 4> expS;
  expS.setIdentity();
  expS.template block<3, 3>(0, 0) = exp;
  expS.template block<3, 1>(0, 3) = jv;
  // expS[4][1] = expS[4][2] = expS[4][3] = 0;
  // expS[4][4] = 1;

  // UNUSED(xi);
  return expS;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 3, 3> R;
  R = mat.template block<3, 3>(0, 0);
  Eigen::Matrix<T, 3, 1> t;
  t = mat.template block<3, 1>(0, 3);
  double theta = acos((R.trace() - 1.0) / 2.0);
  double k = (theta / (2 * sin(theta)));
  Eigen::Matrix<T, 3, 1> w;
  w(0, 0) = (R(2, 1) - R(1, 2)) * k;
  w(1, 0) = (R(0, 2) - R(2, 0)) * k;
  w(2, 0) = (R(1, 0) - R(0, 1)) * k;
  if (theta == 0) {
    w.setZero();
  }
  Eigen::Matrix<T, 3, 3> w_hat;
  w_hat.setZero();
  w_hat(0, 1) = -w(2);
  w_hat(1, 0) = w(2);
  w_hat(0, 2) = w(1);
  w_hat(2, 0) = -w(1);
  w_hat(1, 2) = -w(0);
  w_hat(2, 1) = w(0);
  Eigen::Matrix<T, 3, 3> I;
  I.setIdentity();
  Eigen::Matrix<T, 3, 3> J_inv =
      I - (w_hat / 2.0) +
      ((1.0 / (theta * theta)) -
       ((1.0 + cos(theta)) / (2 * theta * sin(theta)))) *
          (w_hat * w_hat);
  if (theta == 0) {
    J_inv.setIdentity();
  }
  Eigen::Matrix<T, 3, 1> v = J_inv * t;
  Eigen::Matrix<T, 6, 1> log;
  log.template block<3, 1>(0, 0) = v;
  log.template block<3, 1>(3, 0) = w;

  // UNUSED(mat);
  return log;
}

}  // namespace visnav
