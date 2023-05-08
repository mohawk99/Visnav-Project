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

  Eigen::Matrix3d w_hat;
  w_hat.setZero();
  w_hat(0, 1) = xi(2);
  w_hat(1, 0) = -xi(2);
  w_hat(0, 2) = xi(1);
  w_hat(2, 0) = -xi(1);
  w_hat(1, 2) = xi(0);
  w_hat(2, 1) = -xi(0);
  double theta = xi.norm();
  Eigen::Matrix3d expS = Eigen::Matrix3d::Identity() + (w_hat / theta) * sin(theta) + (w_hat * w_hat) / (theta * theta) * (1 - cos(theta)); 


 // UNUSED(xi);
  return expS;
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  float theta = acos((mat[1][1]+mat[2][2]+mat[3][3] - 1)/2);
  float k = (theta/(2*sin(theta)));
  Eigen::Matrix<T, 3, 1> log;
  log[1][1] = (mat[3][2] - mat[2][3])/k;
  log[2][1] = (mat[1][3] - mat[3][1])/k;
  log[3][1] = (mat[2][1] - mat[1][2])/k;
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
  for(i=1; i<4; i++){
    v[i][1] = xi[i][1]; 
  }
  for(i=1; i<4; i++){
    w[i][1] = xi[i+3][1]; 
  }
  Eigen::Matrix3d w_hat;
  w_hat.setZero();
  w_hat(0, 1) = w(2);
  w_hat(1, 0) = -w(2);
  w_hat(0, 2) = w(1);
  w_hat(2, 0) = -w(1);
  w_hat(1, 2) = w(0);
  w_hat(2, 1) = -w(0);
  double theta = w.norm();
  Eigen::Matrix3d exp = Eigen::Matrix3d::Identity() + (w_hat / theta) * sin(theta) + (w_hat * w_hat) / (theta * theta) * (1 - cos(theta));
  Eigen::Matrix3d J = Eigen::Matrix3d::Identity() + w_hat / (theta * theta) * (1 - cos(theta)) + ((w_hat * w_hat)/(theta * theta * theta)) * (theta - sin(theta));
  Eigen::Matrix<T, 3, 1> jv = J * v;
  Eigen::Matrix<T, 4, 4> expS;
  expS.block<3, 3>(0, 0) = exp;
  expS.block<3, 1>(0, 3) = jv;
  exps[4][1] = exps[4][2] = exps[4][3] = 0;
  exps[4][4] = 1;


 // UNUSED(xi);
  return expS;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix3d R = mat.block<3, 3>(0, 0);
  Eigen::Matrix3d t = mat.block<3, 1>(0, 3);
  float theta = acos((R[1][1]+R[2][2]R[3][3] - 1)/2);
  float k = (theta/(2*sin(theta)));
  Eigen::Matrix<T, 3, 1> w;
  w[1][1] = (R[3][2] - R[2][3])/k;
  w[2][1] = (R[1][3] - R[3][1])/k;
  w[3][1] = (R[2][1] - R[1][2])/k;
  Eigen::Matrix3d w_hat;
  w_hat.setZero();
  w_hat(0, 1) = w(2);
  w_hat(1, 0) = -w(2);
  w_hat(0, 2) = w(1);
  w_hat(2, 0) = -w(1);
  w_hat(1, 2) = w(0);
  w_hat(2, 1) = -w(0);
  Eigen::Matrix3d J_inv = Eigen::Matrix3d::Identity() - (w_hat/2) + ((1/(theta*theta)) - (((1 + cos(theta))/(2*theta*sin(theta)))) * (w_hat * w_hat));
  Eigen::Matrix<T, 3, 1> v = J_inv * t;
  Eigen::Matrix<T, 6, 1> log;
  log.block<3, 1>(0, 0) = v;
  log.block<3, 1>(3, 0) = w;

  // UNUSED(mat);
  return log;
}

}  // namespace visnav
