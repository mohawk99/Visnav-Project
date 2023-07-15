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

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

template <class T>
class AbstractCamera;

struct ReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                          const Eigen::Vector3d& p_3d,
                          const std::string& cam_model)
      : p_2d(p_2d), p_3d(p_3d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_i_c,
                  T const* const sIntr, T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_i_c(sT_i_c);

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 2: implement the rest of the functor

    //  Eigen::Vector3d p_c = T_i_c.inverse() * T_w_i.inverse() * p_3d;
    //  Eigen::Vector2d p_c1 = cam->project(p_c);
    //  residuals = p_2d - p_c1;

    residuals = (p_2d)-cam->project(T_i_c.inverse() * T_w_i.inverse() * p_3d);

    return true;
  }

  Eigen::Vector2d p_2d;
  Eigen::Vector3d p_3d;
  std::string cam_model;
};

struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                                          const std::string& cam_model)
      : p_2d(p_2d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_c, T const* const sp_3d_w,
                  T const* const sIntr, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T_w_c(sT_w_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3d_w(sp_3d_w);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 4: Compute reprojection error
    residuals = (p_2d)-cam->project(T_w_c.inverse() * p_3d_w);

    return true;
  }

  Eigen::Vector2d p_2d;
  std::string cam_model;
};


struct PoseGraphCostFunctor {
    PoseGraphCostFunctor(const Sophus::SE3d& measured_relative_pose)
        : measured_relative_pose_(measured_relative_pose) {}

    template<typename T>
    bool operator()(const T* const T_w_0, const T* const T_w_1, T* residuals) const {
        // Convert the input variables (T_w_0 and T_w_1) to Sophus SE3d transformations
        Sophus::SE3<T> se3_T_w_0 = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(T_w_0));
        Sophus::SE3<T> se3_T_w_1 = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(T_w_1));

        // Compute the residual: (T_w_0.inverse() * T_w_1) * measured_relative_pose_.inverse()
        Sophus::SE3<T> residual = (se3_T_w_0.inverse() * se3_T_w_1) * measured_relative_pose_.inverse().template cast<T>();

        // Convert the residual to Lie algebra vector
        Eigen::Matrix<T, 6, 1> residual_lie_algebra = residual.log();

        // Copy the residual to the output array
        for (int i = 0; i < 6; ++i) {
            residuals[i] = residual_lie_algebra[i];
        }

        return true;
    }

    //static ceres::CostFunction* Create(const Sophus::SE3d& measured_relative_pose) {
    //    return new ceres::AutoDiffCostFunction<PoseGraphCostFunctor, 6, 6, 6>(
    //        new PoseGraphCostFunctor(measured_relative_pose)
    //    );
    //}

//private:
    Sophus::SE3d measured_relative_pose_;
};


}  // namespace visnav
