#include <visnav/common_types.h>
#include <iostream>
#include <set>
#include <algorithm>

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

template <typename T>
std::set<T, std::greater<T>> sortSetDescending(const std::set<T>& inputSet) {
  std::set<T, std::greater<T>> sortedSet(inputSet.begin(), inputSet.end());
  return sortedSet;
}

template <typename T>
std::set<T, std::greater<T>> getTopNElements(const std::set<T>& inputSet,
                                             int N) {
  std::set<T, std::greater<T>> sortedSet = sortSetDescending(inputSet);

  // Extract the top N elements
  auto it = sortedSet.begin();
  std::advance(it, N);

  std::set<T, std::greater<T>> topNSet(sortedSet.begin(), it);
  return topNSet;
}

void DrawCameraCenter(const Eigen::Matrix4d& T_w_c, const float& color_r,
                      const float& color_g, const float& color_b) {
  // Extract the camera center from the transformation matrix
  Eigen::Vector3d camera_center = T_w_c.block<3, 1>(0, 3);

  // Set the color for drawing the camera center
  glColor3f(color_r, color_g, color_b);

  // Draw a small point at the camera center
  glPointSize(5);
  glBegin(GL_POINTS);
  glVertex3d(camera_center.x(), camera_center.y(), camera_center.z());
  glEnd();
}

void DrawLineBetweenCameras(const Eigen::Matrix4d& T_w_c1,
                            const Eigen::Matrix4d& T_w_c2, const float& color_r,
                            const float& color_g, const float& color_b) {
  // Extract the camera centers from the transformation matrices
  Eigen::Vector3d camera_center1 = T_w_c1.block<3, 1>(0, 3);
  Eigen::Vector3d camera_center2 = T_w_c2.block<3, 1>(0, 3);

  // Set the color for drawing the line
  glColor3f(color_r, color_g, color_b);

  // Draw a line between the camera centers
  glBegin(GL_LINES);
  glVertex3d(camera_center1.x(), camera_center1.y(), camera_center1.z());
  glVertex3d(camera_center2.x(), camera_center2.y(), camera_center2.z());
  glEnd();
}