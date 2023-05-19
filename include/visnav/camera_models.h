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

template <typename Scalar>
class AbstractCamera;

template <typename Scalar>
class PinholeCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  PinholeCamera() = default;
  PinholeCamera(const VecN& p) : param(p) {}

  static PinholeCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0, 0, 0, 0;
    PinholeCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "pinhole"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    Vec2 res;
    res[0] = fx * (x / z) + cx;
    res[1] = fy * (y / z) + cy;

    // TODO SHEET 2: implement camera model
    // UNUSED(fx);
    // UNUSED(fy);
    // UNUSED(cx);
    // UNUSED(cy);
    // UNUSED(x);
    // UNUSED(y);
    // UNUSED(z);

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& u = p[0];
    const Scalar& v = p[1];

    double mx;
    double my;
    double abs;

    mx = (u - cx)/fx;
    my = (v - cy)/fy;
    abs = 1/(sqrt(mx*mx+my*my+1));

    Vec3 res;
    res[0] = mx*abs;
    res[1] = my*abs;
    res[2] = 1*abs;

    

    // TODO SHEET 2: implement camera model
    // UNUSED(p);
    // UNUSED(fx);
    // UNUSED(fy);
    // UNUSED(cx);
    // UNUSED(cy);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar = double>
class ExtendedUnifiedCamera : public AbstractCamera<Scalar> {
 public:
  // NOTE: For convenience for serialization and handling different camera
  // models in ceres functors, we use the same parameter vector size for all of
  // them, even if that means that for some certain entries are unused /
  // constant 0.
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  ExtendedUnifiedCamera() = default;
  ExtendedUnifiedCamera(const VecN& p) : param(p) {}

  static ExtendedUnifiedCamera getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 500, 0.5 * 500, 319.5, 239.5, 0.51231234, 0.9, 0, 0;
    ExtendedUnifiedCamera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static const std::string getName() { return "eucm"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];
    double k;
    double d;

    Vec2 res;
    d = sqrt(beta*(x*x+y*y)+z*z);
    k = 1/(alpha*d+(1-alpha)*z);
    res[0] = fx*x*k + cx;
    res[1] = fy*y*k + cy;
 
    // TODO SHEET 2: implement camera model
    // UNUSED(fx);
    // UNUSED(fy);
    // UNUSED(cx);
    // UNUSED(cy);
    // UNUSED(alpha);
    // UNUSED(beta);
    // UNUSED(x);
    // UNUSED(y);
    // UNUSED(z);

    return res;
  }

  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];
    const Scalar& beta = param[5];

    Vec3 res;
    const Scalar& u = p[0];
    const Scalar& v = p[1];

    double mx;
    double my;
    double mz;
    double r;
    double abs;

    mx = (u - cx)/fx;
    my = (v - cy)/fy;
    r = sqrt(mx*mx + my*my);
    mz = (1 - beta*alpha*alpha*r*r)/(alpha*sqrt(1 - (2*alpha - 1)*beta*r*r) + 1 - alpha);
    abs = 1/(sqrt(mx*mx+my*my+mz*mz));

    res[0] = mx*abs;
    res[1] = my*abs;
    res[2] = mz*abs;



    // TODO SHEET 2: implement camera model

    UNUSED(p);
    UNUSED(fx);
    UNUSED(fy);
    UNUSED(cx);
    UNUSED(cy);
    UNUSED(alpha);
    UNUSED(beta);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar>
class DoubleSphereCamera : public AbstractCamera<Scalar> {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  DoubleSphereCamera() = default;
  DoubleSphereCamera(const VecN& p) : param(p) {}

  static DoubleSphereCamera<Scalar> getTestProjections() {
    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785, 0,
        0;
    DoubleSphereCamera<Scalar> res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }
  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "ds"; }
  std::string name() const { return getName(); }

  virtual Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    double k;
    double d1;
    double d2;

    d1 = sqrt(x*x + y*y + z*z);
    d2 = sqrt( x*x + y*y + (xi*d1 + z) * (xi*d1 + z));
    k = 1/(alpha*d2 + (1-alpha)*(xi*d1 + z));


    Vec2 res;
    res[0] = fx*x*k + cx;
    res[1] = fy*y*k + cy;

    // TODO SHEET 2: implement camera model
    // UNUSED(fx);
    // UNUSED(fy);
    // UNUSED(cx);
    // UNUSED(cy);
    // UNUSED(xi);
    // UNUSED(alpha);
    // UNUSED(x);
    // UNUSED(y);
    // UNUSED(z);

    return res;
  }

  virtual Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& xi = param[4];
    const Scalar& alpha = param[5];

    double mx;
    double my;
    double mz;
    double r;
    double abs;
    const Scalar& u = p[0];
    const Scalar& v = p[1];

    mx = (u - cx)/fx;
    my = (v - cy)/fy;
    r = sqrt(mx*mx + my*my);
    mz = (1 - alpha*alpha*r*r)/(alpha*sqrt(1 - (2*alpha - 1)*r*r) + 1 - alpha);
    abs = (mz*xi + sqrt(mz*mz+ (1 - xi*xi)*r*r))/(mz*mz + r*r);

    Vec3 res;
    res[0] = mx*abs;
    res[1] = my*abs;
    res[2] = mz*abs - xi;

    // TODO SHEET 2: implement camera model
    // UNUSED(p);
    // UNUSED(fx);
    // UNUSED(fy);
    // UNUSED(cx);
    // UNUSED(cy);
    // UNUSED(xi);
    // UNUSED(alpha);
    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar = double>
class KannalaBrandt4Camera : public AbstractCamera<Scalar> {
 public:
  static constexpr int N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;
  typedef Eigen::Matrix<Scalar, 4, 1> Vec4;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  KannalaBrandt4Camera() = default;
  KannalaBrandt4Camera(const VecN& p) : param(p) {}

  static KannalaBrandt4Camera getTestProjections() {
    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    KannalaBrandt4Camera res(vec1);

    return res;
  }

  Scalar* data() { return param.data(); }

  const Scalar* data() const { return param.data(); }

  static std::string getName() { return "kb4"; }
  std::string name() const { return getName(); }

  inline Vec2 project(const Vec3& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p[0];
    const Scalar& y = p[1];
    const Scalar& z = p[2];

    double r;
    double t;
    double d_t;

    r = sqrt(x*x + y*y);
    t = atan2(r,z);
    d_t = t + k1*t*t*t + k2*t*t*t*t*t + k3*t*t*t*t*t*t*t + k4*t*t*t*t*t*t*t*t*t;

    Vec2 res;
    res[0] = (fx*x*d_t)/r + cx;
    res[1] = (fy*y*d_t)/r + cy; 

    // TODO SHEET 2: implement camera model
    // UNUSED(fx);
    // UNUSED(fy);
    // UNUSED(cx);
    // UNUSED(cy);
    // UNUSED(k1);
    // UNUSED(k2);
    // UNUSED(k3);
    // UNUSED(k4);
    // UNUSED(x);
    // UNUSED(y);
    // UNUSED(z);

    return res;
  }
double f_t(double t, const Scalar& k1, const Scalar& k2, const Scalar& k3, const Scalar& k4) const{
    // return (t + t*t*t + t*t*t*t*t + t*t*t*t*t*t*t + t*t*t*t*t*t*t*t*t); 
    return (t * (1 + t*t*(k1 + t*t*(k2 + t*t*(k3 + k4*t*t))))); 
}

double d_t(double t, const Scalar& k1, const Scalar& k2, const Scalar& k3, const Scalar& k4) const{
    // return (1 + 3*t*t + 5*t*t*t*t + 7*t*t*t*t*t*t + 9*t*t*t*t*t*t*t*t);
    return (1 + t*t*(3*k1 + t*t*(5*k2 + t*t*(7*k3 + 9*k4*t*t))));  
}
  Vec3 unproject(const Vec2& p) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& u = p[0];
    const Scalar& v = p[1];
    double mx;
    double my;
    double r;
    

    mx = (u - cx)/fx;
    my = (v - cy)/fy;
    r = sqrt(mx*mx + my*my);

    double t = M_PI/2;
    double t_next = 0;
    double limit = 1e-14;
    double diff = 0;
  //  double f_t = 0;
  //  double d_t = 0;

    while(true){

        t_next = t - ((f_t(t,k1,k2,k3,k4) - r)/d_t(t,k1,k2,k3,k4));

        if (t_next > t)
        {
            diff = t_next - t;
        }
        else
        {
            diff = t - t_next;
        }

        if(diff < limit)
        {
            break;
        }

        t = t_next;
    }



   //  double f_t = t + k1*t*t*t + k2*t*t*t*t*t + k3*t*t*t*t*t*t*t + k4*t*t*t*t*t*t*t*t*t;
   //  double d_t = 1 + 3*k1*t*t + 5*k2*t*t*t*t + 7*k3**t*t*t*t*t*t + 9*k4**t*t*t*t*t*t*t*t;

 //   for(int i = 0; i < 4; i++){
 //       t -= (f_t(t,k1,k2,k3,k4))/d_t(t,k1,k2,k3,k4);
 //   }


    Vec3 res;
    res[0] = (sin(t)*mx)/r;
    res[1] = (sin(t)*my)/r;
    res[2] = cos(t);

    // TODO SHEET 2: implement camera model
    // UNUSED(p);
    // UNUSED(fx);
    // UNUSED(fy);
    // UNUSED(cx);
    // UNUSED(cy);

    return res;
  }

  const VecN& getParam() const { return param; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param = VecN::Zero();
};

template <typename Scalar>
class AbstractCamera {
 public:
  static constexpr size_t N = 8;

  typedef Eigen::Matrix<Scalar, 2, 1> Vec2;
  typedef Eigen::Matrix<Scalar, 3, 1> Vec3;

  typedef Eigen::Matrix<Scalar, N, 1> VecN;

  virtual ~AbstractCamera() = default;

  virtual Scalar* data() = 0;

  virtual const Scalar* data() const = 0;

  virtual Vec2 project(const Vec3& p) const = 0;

  virtual Vec3 unproject(const Vec2& p) const = 0;

  virtual std::string name() const = 0;

  virtual const VecN& getParam() const = 0;

  inline int width() const { return width_; }
  inline int& width() { return width_; }
  inline int height() const { return height_; }
  inline int& height() { return height_; }

  static std::shared_ptr<AbstractCamera> from_data(const std::string& name,
                                                   const Scalar* sIntr) {
    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(new PinholeCamera<Scalar>(intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);
      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

  // Loading from double sphere initialization
  static std::shared_ptr<AbstractCamera> initialize(const std::string& name,
                                                    const Scalar* sIntr) {
    Eigen::Matrix<Scalar, 8, 1> init_intr;

    if (name == DoubleSphereCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;

      return std::shared_ptr<AbstractCamera>(
          new DoubleSphereCamera<Scalar>(init_intr));
    } else if (name == PinholeCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new PinholeCamera<Scalar>(init_intr));
    } else if (name == KannalaBrandt4Camera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();

      return std::shared_ptr<AbstractCamera>(
          new KannalaBrandt4Camera<Scalar>(init_intr));
    } else if (name == ExtendedUnifiedCamera<Scalar>::getName()) {
      Eigen::Map<Eigen::Matrix<Scalar, 8, 1> const> intr(sIntr);

      init_intr = intr;
      init_intr.template tail<4>().setZero();
      init_intr[4] = 0.5;
      init_intr[5] = 1;

      return std::shared_ptr<AbstractCamera>(
          new ExtendedUnifiedCamera<Scalar>(init_intr));
    } else {
      std::cerr << "Camera model " << name << " is not implemented."
                << std::endl;
      std::abort();
    }
  }

 private:
  // image dimensions
  int width_ = 0;
  int height_ = 0;
};

}  // namespace visnav
