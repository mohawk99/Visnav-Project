#include <visnav/common_types.h>
#include <iostream>
#include <set>
#include <algorithm>

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <chrono>
#include <thread>

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

/** EVALUATION: **/
double alignSVD(const std::vector<int64_t>& filter_t_ns,
                const std::vector<Eigen::Vector3d>& filter_t_w_i,
                const std::vector<int64_t>& gt_t_ns,
                std::vector<Eigen::Vector3d>& gt_t_w_i) {
  std::vector<Eigen::Vector3d> est_associations;
  std::vector<Eigen::Vector3d> gt_associations;

  for (size_t i = 0; i < filter_t_w_i.size(); i++) {
    int64_t t_ns = filter_t_ns[i];

    size_t j;
    for (j = 0; j < gt_t_ns.size(); j++) {
      if (gt_t_ns.at(j) > t_ns) break;
    }
    j--;

    if (j >= gt_t_ns.size() - 1) {
      continue;
    }

    double dt_ns = t_ns - gt_t_ns.at(j);
    double int_t_ns = gt_t_ns.at(j + 1) - gt_t_ns.at(j);

    // Skip if the interval between gt is larger than 100ms
    if (int_t_ns > 1.1e8) continue;

    double ratio = dt_ns / int_t_ns;

    Eigen::Vector3d gt = (1 - ratio) * gt_t_w_i[j] + ratio * gt_t_w_i[j + 1];

    gt_associations.emplace_back(gt);
    est_associations.emplace_back(filter_t_w_i[i]);
  }

  int num_kfs = est_associations.size();

  Eigen::Matrix<double, 3, Eigen::Dynamic> gt, est;
  gt.setZero(3, num_kfs);
  est.setZero(3, num_kfs);

  for (size_t i = 0; i < est_associations.size(); i++) {
    gt.col(i) = gt_associations[i];
    est.col(i) = est_associations[i];
  }

  Eigen::Vector3d mean_gt = gt.rowwise().mean();
  Eigen::Vector3d mean_est = est.rowwise().mean();

  gt.colwise() -= mean_gt;
  est.colwise() -= mean_est;

  Eigen::Matrix3d cov = gt * est.transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
    S(2, 2) = -1;

  Eigen::Matrix3d rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Eigen::Vector3d trans = mean_gt - rot_gt_est * mean_est;

  Sophus::SE3d T_gt_est(rot_gt_est, trans);
  Sophus::SE3d T_est_gt = T_gt_est.inverse();

  for (size_t i = 0; i < gt_t_w_i.size(); i++) {
    gt_t_w_i[i] = T_est_gt * gt_t_w_i[i];
  }

  double error = 0;
  for (size_t i = 0; i < est_associations.size(); i++) {
    est_associations[i] = T_gt_est * est_associations[i];
    Eigen::Vector3d res = est_associations[i] - gt_associations[i];

    error += res.transpose() * res;
  }

  error /= est_associations.size();
  error = std::sqrt(error);

  std::cout << "T_align\n" << T_gt_est.matrix() << std::endl;
  std::cout << "error " << error << std::endl;
  std::cout << "number of associations " << num_kfs << std::endl;

  return error;
}

void parseCSV(const std::string& file_path, std::vector<int64_t>& timestamps,
              std::vector<Eigen::Vector3d>& positions) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << file_path << std::endl;
    return;
  }

  std::string line;
  // Skip the first line (header) as it contains column names
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string field;

    // Parse the timestamp (first column)
    std::getline(ss, field, ',');
    int64_t timestamp = std::stoll(field);
    timestamps.push_back(timestamp);

    // Parse the next three columns as position vector (gt_t_w_i)
    Eigen::Vector3d position;
    for (int i = 0; i < 3; i++) {
      std::getline(ss, field, ',');
      position(i) = std::stod(field);
    }

    positions.push_back(position);
  }
 
void imitateSleep(int seconds) {
  for (int i = 0; i < seconds * 10; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(
        100));  // Sleep for 100 milliseconds (0.1 second)
  }
}

void writeDataToCSV(const std::vector<int64_t>& timestamps,
                    const std::vector<Eigen::Vector3d>& positions,
                    const std::string& file_path) {
  std::ofstream file(file_path);

  if (!file.is_open()) {
    std::cerr << "Error opening the file: " << file_path << std::endl;
    return;
  }

  // Write CSV header
  file << "timestamp,x,y,z\n";

  // Write data for each position and timestamp
  for (size_t i = 0; i < positions.size(); ++i) {
    file << timestamps[i] << "," << positions[i].x() << "," << positions[i].y()
         << "," << positions[i].z() << "\n";
  }

  file.close();
}