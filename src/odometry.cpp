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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <visnav/keypoints.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/vo_utils.h>

#include <visnav/gui_helper.h>
#include <visnav/tracks.h>

#include <visnav/serialization.h>

#include <visnav/utilities.h>

#include <visnav/bow_db.h>
#include <visnav/bow_voc.h>

#include <opencv2/opencv.hpp>

using namespace visnav;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t view_id);
void change_display_to_image(const FrameCamId& fcid);
void draw_scene();
void load_data(const std::string& path, const std::string& calib_path);
bool next_step();
void optimize();
void compute_projections();

Eigen::Vector3d extractLandmarkPosition(FrameId frame_id, Landmarks landmarks);
void SetLandmarkPosition(FrameId frame_id, Landmarks landmarks,
                         Eigen::Vector3d pose);

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

int current_frame = 0;
Sophus::SE3d current_pose;
bool take_keyframe = true;
TrackId next_landmark_id = 0;

std::atomic<bool> opt_running{false};
std::atomic<bool> opt_finished{false};

std::set<FrameId> kf_frames;

std::shared_ptr<std::thread> opt_thread;

/// intrinsic calibration
Calibration calib_cam;
Calibration calib_cam_opt;

/// loaded images
tbb::concurrent_unordered_map<FrameCamId, std::string> images;

/// timestamps for all stereo pairs
std::vector<Timestamp> timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// camera poses in the current map
Cameras cameras;

/// copy of cameras for optimization in parallel thread
Cameras cameras_opt;

/// landmark positions and feature observations in current map
Landmarks landmarks;

/// copy of landmarks for optimization in parallel thread
Landmarks landmarks_opt;

/// landmark positions that were removed from the current map
Landmarks old_landmarks;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images
ImageProjections image_projections;

/**
 * PROJECT:
 * Covisibility Graph containing edges if two Key Frames are covisible
 * TODO: Extend class for loop edge too later
 */

CoVisGraph covis_graph;
int num_keyframes = kf_frames.size();
int WINDOW_SIZE = 3;
const std::string vocab_path = "./data/ORBvoc.cereal";
BowDatabase BOW_DB;
BowVocabulary BOW_VOCAB(vocab_path);

std::map<FrameId, bool> loop_candidates;
const int patience = 3;
int loop_consistency_timeout = patience;
pangolin::Var<double> relative_pose_ransac_thresh("hidden.5pt_thresh", 5e-5,
                                                  1e-10, 1, true);

std::vector<std::pair<FrameId, FrameId>> loop_pairs;

bool SAVE_LOOP_PAIRS = false;

std::vector<Eigen::Vector3d> gt_positions;
std::vector<Eigen::Vector3d> pred_positions;
std::vector<int64_t> gt_timestamps;

/** END_PROJECT: **/

///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel
// by switching the prefix from "ui" to "hidden" or vice verca. This way you
// can show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);

//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);

pangolin::Var<double> match_max_dist_2d("hidden.match_max_dist_2d", 20.0, 1.0,
                                        50);

pangolin::Var<int> new_kf_min_inliers("hidden.new_kf_min_inliers", 80, 1, 200);

pangolin::Var<int> max_num_kfs("hidden.max_num_kfs", 10, 5, 20);

pangolin::Var<double> cam_z_threshold("hidden.cam_z_threshold", 0.1, 1.0, 0.0);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path = "data/V1_01_easy/mav0";
  std::string cam_calib = "opt_calib.json";

  CLI::App app{"Visual odometry."};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: " + cam_calib);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(dataset_path, cam_calib);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    /****** PROJECT: Loop Edge Pair Visualization *********/

    // // Create a new window for frame plotting
    // pangolin::CreateWindowAndBind("Loop Edges", 1280, 480);

    // // Create a view for Loop Edgesting
    // pangolin::View& framePlotView = pangolin::Display("Loop Edges")
    //                                     .SetBounds(0.0, 1.0, 0.0, 1.0)
    //                                     .SetLayout(pangolin::LayoutEqual);

    // // Add the view to the window
    // framePlotView.AddDisplay(pangolin::Display("Loop Edges"));

    // // Set the Loop Edgesting function as the draw function for the view
    // framePlotView[0].extern_draw_function =
    //     std::bind(&draw_image_overlay, std::placeholders::_1, 0);

    // // Create a slider for selecting the frame index
    // pangolin::Var<int> frameSlider("Pair Index", 0, 0, loop_pairs.size() -
    // 1); framePlotView.AddDisplay(frameSlider);

    // // Main event loop for Loop Edgesting window
    // while (!pangolin::ShouldQuit()) {
    //   // Clear the view
    //   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //   // Update the frame index based on the slider value
    //   int frameIndex = frameSlider.Get();
    //   show_frame1 = frameIndex;
    //   show_frame2 = frameIndex;

    //   // Render the view
    //   framePlotView.Render();

    //   // Swap buffers and process events
    //   pangolin::FinishFrame();
    // }

    // // Destroy the Loop Edgesting window
    // pangolin::DestroyWindow("Loop Edges");
    /*****************************************************/

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      draw_scene();

      img_view_display.Activate();

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame1, 0));
          change_display_to_image(FrameCamId(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame2, 0));
          change_display_to_image(FrameCamId(show_frame2, 1));
        }
      }

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame1);
        auto cam_id = static_cast<CamId>(show_cam1);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[0]->SetImage(img);
        } else {
          img_view[0]->Clear();
        }
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame2);
        auto cam_id = static_cast<CamId>(show_cam2);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[1]->SetImage(img);
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  } else {
    // non-gui mode: Process all frames, then exit
    while (next_step()) {
      // nop
    }
  }
  std::cout << "[EVALUATION] Calculating ATE. . . \n";
  parseCSV(dataset_path + "/leica0/data.csv", gt_timestamps, gt_positions);
  double ate_error =
      alignSVD(gt_timestamps, pred_positions, gt_timestamps, gt_positions);
  return 0;
}

// void plotFrames(const std::vector<cv::Mat>& frames,
//                 const std::vector<std::pair<int, int>>& indexTuples) {
//   // Initialize Pangolin
//   pangolin::CreateWindowAndBind("Frame Plot", 640, 480);

//   // Create a view for the first frame
//   pangolin::View& view1 = pangolin::CreateDisplay().SetBounds(
//       0.0, 1.0, pangolin::Attach::Pix(0), pangolin::Attach::Pix(480));

//   // Create a view for the second frame
//   pangolin::View& view2 = pangolin::CreateDisplay().SetBounds(
//       0.0, 1.0, pangolin::Attach::Pix(480), pangolin::Attach::Pix(960));

//   // Create a slider for index selection
//   pangolin::Var<int> indexSlider("ui.index", 0, 0, indexTuples.size() - 1);

//   // Add views and slider to the display
//   pangolin::Display("Frame Plot").AddDisplay(view1);
//   pangolin::Display("Frame Plot").AddDisplay(view2);
//   pangolin::Display("Frame Plot").AddVariable(indexSlider);

//   while (!pangolin::ShouldQuit()) {
//     // Clear the views
//     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//     // Select the index tuple based on the slider value
//     int tupleIndex = indexSlider.Get();
//     if (tupleIndex >= 0 && tupleIndex < indexTuples.size()) {
//       std::pair<int, int> indexTuple = indexTuples[tupleIndex];
//       int index1 = indexTuple.first;
//       int index2 = indexTuple.second;

//       // Select the first frame
//       view1.Activate();
//       glClearColor(0.0, 0.0, 0.0, 0.0);
//       glColor3f(1.0, 1.0, 1.0);

//       // Plot the first frame
//       if (index1 >= 0 && index1 < frames.size()) {
//         cv::Mat frame1 = frames[index1];
//         glDrawPixels(frame1.cols, frame1.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE,
//                      frame1.data);
//       }

//       // Select the second frame
//       view2.Activate();
//       glClearColor(0.0, 0.0, 0.0, 0.0);
//       glColor3f(1.0, 1.0, 1.0);

//       // Plot the second frame
//       if (index2 >= 0 && index2 < frames.size()) {
//         cv::Mat frame2 = frames[index2];
//         glDrawPixels(frame2.cols, frame2.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE,
//                      frame2.data);
//       }
//     }

//     // Swap frames and process events
//     pangolin::FinishFrame();
//   }

//   // Close the window
//   pangolin::DestroyWindow("Frame Plot");
// }

// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  auto frame_id =
      static_cast<FrameId>(view_id == 0 ? show_frame1 : show_frame2);
  auto cam_id = static_cast<CamId>(view_id == 0 ? show_cam1 : show_cam2);

  FrameCamId fcid(frame_id, cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(fcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(fcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, text_row);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }

  if (show_reprojections) {
    if (image_projections.count(fcid) > 0) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);  // red
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      const size_t num_points = image_projections.at(fcid).obs.size();
      double error_sum = 0;
      size_t num_outliers = 0;

      // count up and draw all inlier projections
      for (const auto& lm_proj : image_projections.at(fcid).obs) {
        error_sum += lm_proj->reprojection_error;

        if (lm_proj->outlier_flags != OutlierNone) {
          // outlier point
          glColor3f(1.0, 0.0, 0.0);  // red
          ++num_outliers;
        } else if (lm_proj->reprojection_error >
                   reprojection_error_huber_pixel) {
          // close to outlier point
          glColor3f(1.0, 0.5, 0.0);  // orange
        } else {
          // clear inlier point
          glColor3f(1.0, 1.0, 0.0);  // yellow
        }
        pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
        pangolin::glDrawLine(lm_proj->point_measured,
                             lm_proj->point_reprojected);
      }

      // only draw outlier projections
      if (show_outlier_observations) {
        glColor3f(1.0, 0.0, 0.0);  // red
        for (const auto& lm_proj : image_projections.at(fcid).outlier_obs) {
          pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
          pangolin::glDrawLine(lm_proj->point_measured,
                               lm_proj->point_reprojected);
        }
      }

      glColor3f(1.0, 0.0, 0.0);  // red
      pangolin::GlFont::I()
          .Text("Average repr. error (%u points, %u new outliers): %.2f",
                num_points, num_outliers, error_sum / num_points)
          .Draw(5, text_row);
      text_row += 20;
    }
  }

  if (show_epipolar) {
    glLineWidth(1.0);
    glColor3f(0.0, 1.0, 1.0);  // bright teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && it->second.inliers.size() > 20) {
      Sophus::SE3d T_this_other =
          idx == 0 ? it->second.T_i_j : it->second.T_i_j.inverse();

      Eigen::Vector3d p0 = T_this_other.translation().normalized();

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector3d p1(0, sin(i), cos(i));

        if (idx == 0) p1 = it->second.T_i_j * p1;

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          line.emplace_back(calib_cam.intrinsics[cam_id]->project(
              p0 * j + (1 - std::abs(j)) * p1));
        }

        Eigen::Vector2d c = calib_cam.intrinsics[cam_id]->project(p1);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }
}

// Update the image views to a given image id
void change_display_to_image(const FrameCamId& fcid) {
  if (0 == fcid.cam_id) {
    // left view
    show_cam1 = 0;
    show_frame1 = fcid.frame_id;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = fcid.cam_id;
    show_frame2 = fcid.frame_id;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {
  const FrameCamId fcid1(show_frame1, show_cam1);
  const FrameCamId fcid2(show_frame2, show_cam2);

  const u_int8_t color_camera_current[3]{255, 0, 0};         // red
  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_old_points[3]{170, 170, 170};         // gray
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

  // render cameras
  if (show_cameras3d) {
    for (const auto& cam : cameras) {
      if (cam.first == fcid1) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                      0.1f);
      } else if (cam.first == fcid2) {
        render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_right,
                      0.1f);
      } else if (cam.first.cam_id == 0) {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left, 0.1f);
      } else {
        render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_right,
                      0.1f);
      }
    }
    render_camera(current_pose.matrix(), 2.0f, color_camera_current, 0.1f);
    const Eigen::Matrix4d& T_w_c2 = current_pose.matrix();

    // PLOT FOR ALL KF_FRAMES
    for (const auto& kv : covis_graph.edges) {
      FrameId kf = kv.first;
      std::vector<GraphEdge> edges = kv.second;

      const Eigen::Matrix4d& T_w_c1 = covis_graph.poses[kf];

      for (auto e : edges) {
        FrameId frame = e.value;
        int type = e.type;
        float weight = e.weight;

        const Eigen::Matrix4d& T_w_c2 = covis_graph.poses[frame];

        DrawCameraCenter(T_w_c1, 1.0f, 0.0f,
                         0.0f);  // Red color for camera 1
        DrawCameraCenter(T_w_c2, 0.0f, 0.0f,
                         1.0f);  // Blue color for camera 2

        u_int8_t color_line[3];

        if (type == 1) {
          DrawLineBetweenCameras(T_w_c1, T_w_c2, 0.0f, 1.0f, 0.0f);
        } else {
          DrawLineBetweenCameras(T_w_c1, T_w_c2, 0.0f, 0.0f, 1.0f);
        }
      }
    }
  }

  // render points
  if (show_points3d && landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& kv_lm : landmarks) {
      const bool in_cam_1 = kv_lm.second.obs.count(fcid1) > 0;
      const bool in_cam_2 = kv_lm.second.obs.count(fcid2) > 0;

      const bool outlier_in_cam_1 = kv_lm.second.outlier_obs.count(fcid1) > 0;
      const bool outlier_in_cam_2 = kv_lm.second.outlier_obs.count(fcid2) > 0;

      if (in_cam_1 && in_cam_2) {
        glColor3ubv(color_selected_both);
      } else if (in_cam_1) {
        glColor3ubv(color_selected_left);
      } else if (in_cam_2) {
        glColor3ubv(color_selected_right);
      } else if (outlier_in_cam_1 || outlier_in_cam_2) {
        glColor3ubv(color_outlier_observation);
      } else {
        glColor3ubv(color_points);
      }

      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }

  // render points
  if (show_old_points3d && old_landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);

    for (const auto& kv_lm : old_landmarks) {
      glColor3ubv(color_old_points);
      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }
}

// Load images, calibration, and features / matches if available
void load_data(const std::string& dataset_path, const std::string& calib_path) {
  const std::string timestams_path = dataset_path + "/cam0/data.csv";

  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;

      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        timestamps.push_back(timestamp);
      }

      std::string img_name = line.substr(20, line.size() - 21);

      for (int i = 0; i < NUM_CAMS; i++) {
        FrameCamId fcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[fcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);
      std::cout << "Loaded camera from " << calib_path << " with models ";
      for (const auto& cam : calib_cam.intrinsics) {
        std::cout << cam->name() << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Execute next step in the overall odometry pipeline. Call this repeatedly
// until it returns false for automatic execution.
bool next_step() {
  if (current_frame >= int(images.size()) / NUM_CAMS) return false;

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  if (take_keyframe) {
    take_keyframe = false;

    FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "KF Projected " << projected_track_ids.size() << " points."
              << std::endl;

    MatchData md_stereo;
    KeypointsData kdl, kdr;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);
    pangolin::ManagedImage<uint8_t> imgr = pangolin::LoadImage(images[fcidr]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);
    detectKeypointsAndDescriptors(imgr, kdr, num_features_per_image,
                                  rotate_features);

    md_stereo.T_i_j = T_0_1;

    Eigen::Matrix3d E;
    computeEssential(T_0_1, E);

    matchDescriptors(kdl.corner_descriptors, kdr.corner_descriptors,
                     md_stereo.matches, feature_match_max_dist,
                     feature_match_test_next_best);

    findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md_stereo);

    std::cout << "KF Found " << md_stereo.inliers.size() << " stereo-matches."
              << std::endl;

    feature_corners[fcidl] = kdl;
    feature_corners[fcidr] = kdr;
    feature_matches[std::make_pair(fcidl, fcidr)] = md_stereo;

    LandmarkMatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "KF Found " << md.matches.size() << " matches." << std::endl;

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    current_pose = md.T_w_c;

    cameras[fcidl].T_w_c = current_pose;
    cameras[fcidr].T_w_c = current_pose * T_0_1;

    add_new_landmarks(fcidl, fcidr, kdl, kdr, calib_cam, md_stereo, md,
                      landmarks, next_landmark_id);

    remove_old_keyframes(fcidl, max_num_kfs, cameras, landmarks, old_landmarks,
                         kf_frames);
    optimize();

    current_pose = cameras[fcidl].T_w_c;

    /******* !!!!!!!!!!!!! COVIS LOGIC !!!!!!!!!!!!!!! ***********/

    /**
     * Add Covisibility Edges here
     */

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    MatchData md1;
    Sophus::SE3d ckf_T;
    Sophus::SE3d abs_pose1;
    Sophus::SE3d abs_pose2;

    const int MATCH_THRESHOLD = 70;
    const double DIST_2_BEST = 1.2;
    bool new_keyframe_added = kf_frames.size() > num_keyframes;
    auto covis_candidates = getTopNElements(kf_frames, WINDOW_SIZE + 1);

    FrameId ckf = *covis_candidates.begin();

    // Add cameras for plotting
    covis_graph.poses[ckf] = cameras[FrameCamId(ckf, 0)].T_w_c.matrix();

    if (new_keyframe_added && covis_candidates.size() > 0) {
      KeypointsData kdl = feature_corners[FrameCamId(ckf, 0)];

      // Add Bow vector
      BowVector kf_bow;
      BOW_VOCAB.transform(kdl.corner_descriptors, kf_bow);
      BOW_DB.insert(FrameCamId(ckf, 0), kf_bow);

      // std::cout << "New Keyframe Added: " << ckf << "\n";

      // std::cout << "Current KeyFrame: " << ckf << " | Candidate KeyFrames"
      // << "(" << covis_candidates.size() << "): ";

      for (auto& candidate_kf : covis_candidates) {
        if (candidate_kf == ckf) continue;
        // std::cout << candidate_kf << ", ";
        KeypointsData kdl_candidate =
            feature_corners[FrameCamId(candidate_kf, 0)];

        // Add BoW for candidate keyframe
        BowVector cand_kf_bow;
        BOW_VOCAB.transform(kdl_candidate.corner_descriptors, cand_kf_bow);
        BOW_DB.insert(FrameCamId(candidate_kf, 0), cand_kf_bow);

        // std::vector<std::pair<FeatureId, FeatureId>> desc_matches;
        MatchData ransac_md;
        matchDescriptors(kdl_candidate.corner_descriptors,
                         kdl.corner_descriptors, ransac_md.matches,
                         MATCH_THRESHOLD, DIST_2_BEST);

        // Use brute-force matching for covisibility
        int min_matches = 20;
        bool covis = ransac_md.matches.size() > min_matches ? true : false;

        if (covis) {
          GraphEdge covis_edge;
          covis_edge.type = 1;
          covis_edge.weight = ransac_md.matches.size();
          covis_edge.value = candidate_kf;
          covis_edge.desc_matches = ransac_md.matches;
          std::cout << "Adding Covis Edge from " << ckf << " to "
                    << candidate_kf << "\n";
          covis_graph.add_edge(ckf, covis_edge);
        }
      }
      // Covis Consistency check
      if (covis_candidates.size() > 1) {
        FrameId prev_frame = *std::next(covis_candidates.begin(), 1);
        std::cout << "Current Frame: " << ckf
                  << " | Previous Frame: " << prev_frame << "\n";
        auto ckf_covis_edge = covis_graph.find_edge(ckf, prev_frame);
        if (ckf_covis_edge.value == -1) {
          GraphEdge e;
          e.type = 1;
          e.value = prev_frame;

          covis_graph.add_edge(ckf, e);
        }
      }
      // std::cout << "\n";

      /** TODO: Loop Candidate Selection*/
      int keeptopk = 3;
      int theta_min = 30;
      auto neighbours = covis_graph.find_neighbours(ckf, keeptopk);

      BowQueryResult query_result;
      BOW_DB.query(kf_bow, theta_min + 1,
                   query_result);  // + 1 because selfmatch will be discarded
      query_result.erase(query_result.begin());  // Discard self match

      // Use a variant of ORB-SLAM paper
      double min_score = -1;
      for (auto kv : query_result) {
        FrameId fid = kv.first.frame_id;
        double score = kv.second;

        bool is_neighbour = std::find(neighbours.begin(), neighbours.end(),
                                      fid) != neighbours.end();

        if (is_neighbour) {
          if (min_score == -1 || score <= min_score) {
            min_score = score;
          }
        }
      }
      std::vector<FrameId> filtered_candidates;

      for (auto kv : query_result) {
        FrameId fid = kv.first.frame_id;
        double score = kv.second;

        bool is_direct_edge =
            std::find(covis_candidates.begin(), covis_candidates.end(), fid) !=
            covis_candidates.end();

        if (score <= min_score && !is_direct_edge) {
          if (loop_candidates.find(fid) != loop_candidates.end()) {
            loop_candidates[fid] &= true;
          } else {
            loop_candidates[fid] = true;
          }
        } else {
          if (loop_candidates.find(fid) != loop_candidates.end()) {
            loop_candidates[fid] &= false;
          }
        }
      }
      loop_consistency_timeout--;
      if (loop_consistency_timeout == 0) {
        loop_consistency_timeout = patience;
        std::vector<FrameId> keysToErase;
        for (auto kv : loop_candidates) {
          FrameId fid = kv.first;
          bool is_consistent = kv.second;
          if (!is_consistent) {
            if (loop_candidates.find(fid) != loop_candidates.end()) {
              keysToErase.push_back(fid);
            }
          }
        }

        for (const FrameId& fid : keysToErase) {
          loop_candidates.erase(fid);
        }
      }
    }
    /**
     *  From loop candidates detect loop
     *  1) Find similarity transform between current KF and loop candidates
     *  2) If inliers < threshold, then accept the loop candidate and discard
     *the others 3) Convert the loop candidate to covisibility edge
     **/

    int inlier_threshold = 20;  // For RANSAC inliers
    std::vector<FrameId> accepted_loop_cands;
    for (auto kv : loop_candidates) {
      FrameId fid = kv.first;
      GraphEdge edge = covis_graph.find_edge(ckf, fid);

      auto kdl_candidate = feature_corners[FrameCamId(fid, 0)];
      MatchData loop_md;
      // loop_md.matches = edge.desc_matches;

      matchDescriptors(kdl.corner_descriptors, kdl_candidate.corner_descriptors,
                       loop_md.matches, MATCH_THRESHOLD, DIST_2_BEST);
      // std::cout << " Matches for Loop Candidate: " << fid << " --> "
      //           << loop_md.matches.size() << "\n";
      findInliersRansac(kdl, kdl_candidate, calib_cam.intrinsics[0],
                        calib_cam.intrinsics[0], relative_pose_ransac_thresh,
                        inlier_threshold, loop_md);

      if (loop_md.inliers.size() > inlier_threshold) {
        GraphEdge loop_edge;
        loop_edge.type = 2;
        loop_edge.value = fid;
        std::cout << "Adding Loop Edge from " << ckf << " to " << fid << "\n";
        covis_graph.add_edge(ckf, loop_edge);

        Node node1, node2;
        Edge poseEdge;
        node1.id = ckf;
        Sophus::SE3d se1(covis_graph.poses[ckf]);
        node1.pose = se1;
        nodes.push_back(node1);

        node2.id = fid;
        Sophus::SE3d se2(covis_graph.poses[fid]);
        node2.pose = se2;
        nodes.push_back(node2);

        poseEdge.id1 = ckf;
        poseEdge.id2 = fid;
        poseEdge.T = node1.pose.inverse() * node2.pose;
        poseEdge.T = md1.T_i_j;
        edges.push_back(poseEdge);

        std::cout << "Nodes and Edges for PGO added"
                  << "\n";

        /** LOOP_EDGE_CONSISTENCY: Add loop edge with neighbours of current
         * frame**/
        auto ckf_covis_frames = covis_graph.getCovisFrames(ckf);
        for (auto ckf_covis_fid : ckf_covis_frames) {
          GraphEdge covis_loop_edge;
          covis_loop_edge.type = 2;
          covis_loop_edge.value = fid;

          covis_graph.add_edge(ckf_covis_fid.value, covis_loop_edge);
        }

        /** FEATURE: writing loop pairs to disk**/
        if (SAVE_LOOP_PAIRS) {
          auto ckf_image = cv::imread(images[FrameCamId(ckf, 0)]);
          auto fid_image = cv::imread(images[FrameCamId(fid, 0)]);

          // Check if the images were loaded successfully
          if (ckf_image.empty() || fid_image.empty()) {
            std::cout << "Failed to load the images." << std::endl;
            return -1;
          }

          // Resize the images to have the same height
          cv::Size targetSize(ckf_image.cols + fid_image.cols, ckf_image.rows);
          // cv::resize(ckf_image, ckf_image, targetSize);
          // cv::resize(fid_image, fid_image, targetSize);

          // Create a new image to hold the merged result
          cv::Mat mergedImage(targetSize.height, targetSize.width,
                              ckf_image.type());

          // Copy the first image to the left side of the merged image
          cv::Rect roi1(cv::Rect(0, 0, ckf_image.cols, ckf_image.rows));
          cv::Mat roickf_image(mergedImage, roi1);
          ckf_image.copyTo(roickf_image);

          // Copy the second image to the right side of the merged image
          cv::Rect roi2(
              cv::Rect(ckf_image.cols, 0, fid_image.cols, fid_image.rows));
          cv::Mat roifid_image(mergedImage, roi2);
          fid_image.copyTo(roifid_image);

          // Save the merged image
          cv::imwrite("data/loop_pairs/" + std::to_string(ckf) + "_" +
                          std::to_string(fid) + ".jpg",
                      mergedImage);
        }
        // Also add covisibility edges
        if (covis_graph.exists(fid)) {
          auto loop_covis_frames = covis_graph.getCovisFrames(fid);
          for (auto loop_covis_kf : loop_covis_frames) {
            covis_graph.add_edge(ckf, loop_covis_kf);
            /*** QUESTION: Do I need to add the other way around too? */
          }
        }
        accepted_loop_cands.push_back(fid);
      }
    }

    std::vector<Node> sortedNodes(nodes);
    std::sort(sortedNodes.begin(), sortedNodes.end(),
              [](const Node& node1, const Node& node2) {
                return node1.id < node2.id;
              });

    const int opt_window = 3;
    ceres::Problem problem;
    Sophus::SE3d multi_T;
    int edges_connected[opt_window] = {0};
    Sophus::SE3d delta_T;

    // Iterate through the nodes
    for (std::size_t i = 0;
         i < (sortedNodes.size() - 1) && sortedNodes.size() > 0; ++i) {
      const Node& current_node = sortedNodes[i];
      const int& node_id1 = current_node.id;
      abs_pose1 = current_node.pose;
      edges_connected[0] = node_id1;

      // See the next nodes to which it has loop edges with
      for (std::size_t j = i + 1;
           j < (sortedNodes.size() - 1) && j <= i + opt_window; ++j) {
        const Node& next_node = sortedNodes[j];
        const int& node_id2 = next_node.id;
        abs_pose2 = next_node.pose;
        edges_connected[j] = node_id2;

        for (const auto& edge : edges) {
          if ((edge.id1 == node_id1 && edge.id2 == node_id2) ||
              (edge.id1 == node_id2 && edge.id2 == node_id1)) {
            // const Eigen::Matrix4d& relative_T = edge.T;
            Sophus::SE3d relative_T = edge.T;

            for (std::size_t k = 0; k < opt_window && edges_connected[k] != 0;
                 ++k) {
              for (const auto& edge1 : edges) {
                if ((edge1.id1 == edges_connected[k + 1] &&
                     edge1.id2 == edges_connected[k]) ||
                    (edge1.id1 == edges_connected[k] &&
                     edge1.id2 == edges_connected[k + 1])) {
                  multi_T = multi_T * edge1.T;
                }
              }
            }

            // const Eigen::Matrix4d& delta_T = relative_T - multi_T;
            Sophus::SE3d::Tangent lie_algebra_1 = relative_T.log();
            Sophus::SE3d::Tangent lie_algebra_2 = multi_T.log();
            Sophus::SE3d::Tangent delta_lie_algebra =
                lie_algebra_1 - lie_algebra_2;
            delta_T = Sophus::SE3d::exp(delta_lie_algebra);

            // Optimization
            problem.AddParameterBlock(abs_pose1.data(), 6);
            problem.AddParameterBlock(abs_pose2.data(), 6);

            problem.AddParameterBlock(delta_T.data(), 6);
            problem.SetParameterBlockConstant(delta_T.data());

            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<PoseGraphCostFunctor, 6, 6, 6>(
                    new PoseGraphCostFunctor(delta_T));
            problem.AddResidualBlock(cost_function, NULL, abs_pose1.data(),
                                     abs_pose2.data());

            ceres::Solver::Options options;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
          }

          // Landmark pose
          Eigen::Vector3d l1_world =
              extractLandmarkPosition(node_id1, landmarks);
          Eigen::Vector3d l1_cam =
              cameras[FrameCamId(node_id1, 0)].T_w_c.inverse() * l1_world;
          Eigen::Vector3d l1_cam_new = delta_T * l1_cam;

          Eigen::Vector3d l2_world =
              extractLandmarkPosition(node_id2, landmarks);
          Eigen::Vector3d l2_cam =
              cameras[FrameCamId(node_id2, 0)].T_w_c.inverse() * l2_world;
          Eigen::Vector3d l2_cam_new = delta_T * l1_cam;

          // Set the optimized poses
          cameras[FrameCamId(node_id1, 0)].T_w_c = abs_pose1;
          cameras[FrameCamId(node_id2, 0)].T_w_c = abs_pose2;

          // Set poses for right camera
          cameras[FrameCamId(node_id1, 1)].T_w_c = abs_pose1 * T_0_1;
          cameras[FrameCamId(node_id2, 1)].T_w_c = abs_pose2 * T_0_1;

          std::cout << "Poses Updated"
                    << "\n";

          // Landmark pose update
          Eigen::Vector3d l1_world_new = abs_pose1 * l1_cam_new;
          SetLandmarkPosition(node_id1, landmarks, l1_world_new);
          Eigen::Vector3d l2_world_new = abs_pose2 * l2_cam_new;
          SetLandmarkPosition(node_id2, landmarks, l2_world_new);

          std::cout << "Landmarks updated"
                    << "\n";
        }
      }
    }

    /** LOOPCLOSURE: **/
    // if (!opt_running && opt_finished) {
    //   for (auto loop_fid : accepted_loop_cands) {
    //     // Move all of landmark observations from current frame to the loop
    //     // candidate
    //     for (auto& lm : landmarks) {
    //       auto track_id = lm.first;
    //       auto landmark = lm.second;
    //       auto lm_obs = landmark.obs;

    //       // Find the observations in the current KF.
    //       if (lm_obs.find(FrameCamId(ckf, 0)) != lm_obs.end() &&
    //           lm_obs.find(FrameCamId(ckf, 1)) != lm_obs.end()) {
    //         auto current_obs_left = lm_obs[FrameCamId(ckf, 0)];
    //         auto current_obs_right = lm_obs[FrameCamId(ckf, 1)];

    //         lm.second.obs[FrameCamId(loop_fid, 0)] = current_obs_left;
    //         lm.second.obs[FrameCamId(loop_fid, 1)] = current_obs_right;

    //         lm.second.obs.erase(FrameCamId(ckf, 0));
    //         lm.second.obs.erase(FrameCamId(ckf, 1));
    //       }
    //     }
    //     optimize();  // Call BA with updated poses from PGO and landmarks
    //     from
    //                  // LoopClosure
    //   }
    // }
    if (!opt_running && opt_finished) {
      for (auto loop_fid : accepted_loop_cands) {
        // Move shared landmark observations from current keyframe to the loop
        // candidate
        for (auto& lm : landmarks) {
          auto track_id = lm.first;
          auto& landmark =
              lm.second;  // Use a reference to directly modify the landmark

          // Check if the landmark has observations in both ckf and loop_fid
          // with the same FeatureId
          auto it_ckf_left = landmark.obs.find(FrameCamId(ckf, 0));
          auto it_ckf_right = landmark.obs.find(FrameCamId(ckf, 1));
          auto it_loop_left = landmark.obs.find(FrameCamId(loop_fid, 0));
          auto it_loop_right = landmark.obs.find(FrameCamId(loop_fid, 1));

          if (it_ckf_left != landmark.obs.end() &&
              it_ckf_right != landmark.obs.end() &&
              it_loop_left != landmark.obs.end() &&
              it_loop_right != landmark.obs.end() &&
              it_ckf_left->second == it_loop_left->second &&
              it_ckf_right->second == it_loop_right->second) {
            // Move the observations from ckf to loop_fid landmark
            auto current_obs_left = it_ckf_left->second;
            auto current_obs_right = it_ckf_right->second;

            landmark.obs[FrameCamId(loop_fid, 0)] = current_obs_left;
            landmark.obs[FrameCamId(loop_fid, 1)] = current_obs_right;

            // Remove the observations from ckf landmark
            landmark.obs.erase(it_ckf_left);
            landmark.obs.erase(it_ckf_right);
          }
        }
        optimize();  // Call BA with updated poses from PGO and landmarks from
                     // LoopClosure
      }
    }

    /***********************************************************/

    // Update camera trajectories vector
    pred_positions.push_back(cameras[FrameCamId(ckf, 0)].T_w_c.translation());

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    compute_projections();

    current_frame++;
    return true;
  } else {
    FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "Projected " << projected_track_ids.size() << " points."
              << std::endl;

    KeypointsData kdl;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);

    feature_corners[fcidl] = kdl;

    LandmarkMatchData md;
    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "Found " << md.matches.size() << " matches." << std::endl;

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    current_pose = md.T_w_c;

    if (int(md.inliers.size()) < new_kf_min_inliers && !opt_running &&
        !opt_finished) {
      take_keyframe = true;
    }

    if (!opt_running && opt_finished) {
      opt_thread->join();
      landmarks = landmarks_opt;
      cameras = cameras_opt;
      calib_cam = calib_cam_opt;

      opt_finished = false;
    }

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    current_frame++;
    return true;
  }
}

Eigen::Vector3d extractLandmarkPosition(FrameId frame_id, Landmarks landmarks) {
  for (const auto& landmark : landmarks) {
    Landmark landmarkData = landmark.second;
    auto it = landmarkData.obs.find(FrameCamId(frame_id, 0));
    if (it != landmarkData.obs.end()) {
      return landmarkData.p;
    }
  }
}
void SetLandmarkPosition(FrameId frame_id, Landmarks landmarks,
                         Eigen::Vector3d pose) {
  for (const auto& landmark : landmarks) {
    Landmark landmarkData = landmark.second;
    auto it = landmarkData.obs.find(FrameCamId(frame_id, 0));
    if (it != landmarkData.obs.end()) {
      landmarkData.p = pose;
    }
  }
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();

  for (const auto& kv_lm : landmarks) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].obs.push_back(proj_lm);
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Optimize the active map with bundle adjustment
void optimize() {
  size_t num_obs = 0;
  for (const auto& kv : landmarks) {
    num_obs += kv.second.obs.size();
  }

  std::cerr << "Optimizing map with " << cameras.size() << " cameras, "
            << landmarks.size() << " points and " << num_obs << " observations."
            << std::endl;

  // Fix oldest two cameras to fix SE3 and scale gauge. Making the whole second
  // camera constant is a bit suboptimal, since we only need 1 DoF, but it's
  // simple and the initial poses should be good from calibration.
  // std::cout << kf_frames.size() << "\n";
  FrameId fid = *(kf_frames.begin());

  // std::cout << "fid " << fid << std::endl;

  // Prepare bundle adjustment
  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = ba_verbose;

  calib_cam_opt = calib_cam;
  cameras_opt = cameras;
  landmarks_opt = landmarks;

  opt_running = true;

  opt_thread.reset(new std::thread([fid, ba_options] {
    std::set<FrameCamId> fixed_cameras = {{fid, 0}, {fid, 1}};

    bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam_opt,
                      cameras_opt, landmarks_opt);
    opt_finished = true;
    opt_running = false;
  }));

  // Update project info cache
  compute_projections();
}
