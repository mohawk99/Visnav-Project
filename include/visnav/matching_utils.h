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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix

  Eigen::Vector3d t = t_0_1.normalized();
  Eigen::Matrix3d t_hat = Sophus::SO3d::hat(t);
  E = t_hat * R_0_1;

  // UNUSED(E);
  // UNUSED(t_0_1);
  // UNUSED(R_0_1);
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers

    Eigen::Vector3d p0 = cam1->unproject(p0_2d);
    Eigen::Vector3d p1 = cam2->unproject(p1_2d);
    double epi = p0.transpose() * E * p1;

    if (((epi) < epipolar_error_threshold) &&
        ((epi) > (-epipolar_error_threshold))) {
      md.inliers.push_back(
          std::make_pair(md.matches[j].first, md.matches[j].second));
    }
    // UNUSED(cam1);
    // UNUSED(cam2);
    // UNUSED(E);
    // UNUSED(epipolar_error_threshold);
    // UNUSED(p0_2d);
    // UNUSED(p1_2d);
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.

  opengv::bearingVectors_t bear1;
  opengv::bearingVectors_t bear2;

  for (long unsigned int i = 0; i < md.matches.size(); i++) {
    bear1.push_back(cam1->unproject(kd1.corners[md.matches[i].first]));
    bear2.push_back(cam2->unproject(kd2.corners[md.matches[i].second]));
  }
  opengv::relative_pose::CentralRelativeAdapter adapter(bear1, bear2);
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter, opengv::sac_problems::relative_pose::
                               CentralRelativePoseSacProblem::NISTER));
  // 0.92 > 0.99
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();

  opengv::transformation_t optimized =
      opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
  ransac.sac_model_->selectWithinDistance(optimized, ransac_thresh,
                                          ransac.inliers_);
  Eigen::Matrix4d res;
  res.block<3, 3>(0, 0) = optimized.block<3, 3>(0, 0);
  res.block<3, 1>(0, 3) = optimized.block<3, 1>(0, 3).normalized();
  res.block<1, 4>(3, 0) = Eigen::Vector4d(0, 0, 0, 1);
  md.T_i_j = Sophus::SE3d(res);

  if ((int)ransac.inliers_.size() > ransac_min_inliers) {
    for (long unsigned int i = 0; i < ransac.inliers_.size(); i++) {
      md.inliers.push_back(md.matches[ransac.inliers_[i]]);
    }
  }

  // UNUSED(kd1);
  // UNUSED(kd2);
  // UNUSED(cam1);
  // UNUSED(cam2);
  // UNUSED(ransac_thresh);
  // UNUSED(ransac_min_inliers);
}
}  // namespace visnav
