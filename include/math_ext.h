#ifndef __MATH_EXT_H_
#define __MATH_EXT_H_

#ifndef PI
#define PI 3.14159265359
#endif

#include <vector>
#include <string>
#include <time.h>
#include <cstdlib>
#include <limits>
#include <map>
#include <fstream>
#include <algorithm>

#include <host_defines.h>
#include <functional.inl>

namespace ext {
  // ========================
  // ===== Save as File =====
  // ========================
  template <typename T>
  void save(const std::vector<T>& v, std::string filename) {
    std::ofstream fs(filename.c_str());

    fs.precision(6);
    fs << std::scientific;
    for (size_t i=0; i<v.size(); ++i)
      fs << v[i] << std::endl;

    fs.close();
  }

  // ==========================
  // ===== Load from File =====
  // ==========================
  template <typename T>
  void load(std::vector<T>& v, std::string filename) {
    v.clear();

    std::ifstream fs(filename.c_str());

    T t;
    while (fs >> t) 
      v.push_back(t);

    fs.close();
  }

  // =================================
  // ===== Summation over Vector =====
  // =================================
  template <typename T>
  T sum(const std::vector<T>& v) {
    T s = 0;
    for (size_t i=0; i<v.size(); ++i)
      s += v[i];
    return s;
  }

  // ==================================
  // ===== First Order Difference =====
  // ==================================
  template <typename T>
  std::vector<T> diff1st(const std::vector<T>& v) {
    std::vector<T> diff(v.size() - 1);
    for (size_t i=0; i<diff.size(); ++i)
      diff[i] = v[i+1] - v[i];
    return diff;
  }

  template <typename T>
  inline bool is_inf(T x) {
    return x == std::numeric_limits<T>::infinity() || x == -std::numeric_limits<T>::infinity();
  }

  // =============================================
  // ===== Normal Distribution Random Number =====
  // =============================================
  template <typename T>
  T unif(T &seed) {
    T a1=3972.0,a2=4094.0;
    T m=2147483647.0;
    T seed1,seed2;
    seed1 = a1 * seed ;
    seed2 = a2 * seed ;
    /* control seed < 10^10 */
    seed1 = seed1 - (long)(seed1/m) * m ;
    seed2 = seed2 - (long)(seed2/m) * m ;
    seed = seed1 * 100000.0 + seed2;
    seed = seed - (long)(seed/m)*m;

    T u = seed/m;
    return (u < 0) ? -u : u;
  }

  template <typename T>
  T randn(T mean, T var) {
    T sz=0.0,v1,v2,sigma,ans;
    T seed1,seed2;
    sigma=sqrt(var);
    seed1=rand() /*+ clock()*123*/;
    v1=unif<T>(seed1);

    do {
      seed2=fabs(/*clock()*1236*var+*/ rand()-seed1);
      v2=unif<T>(seed2);
    } while (v2 == 0);

    sz=cos(2.*PI*v1)*sqrt(-2.*log(v2));
    ans=sz*sigma+mean;
    return(ans);
  }

  template <typename T>
  std::vector<T> randn(size_t size) {
    std::vector<T> v(size);

    for (size_t i=0; i<v.size(); ++i)
      v[i] = randn<T>(0, 1);

    return v;
  }

  // ==========================
  // ===== Uniform Random =====
  // ==========================
  template <typename T>
  T rand01() {
    return (T) ::rand() / (T) RAND_MAX;
  }

  template <typename T>
  std::vector<T> rand(size_t size) {
    std::vector<T> v(size);

    for (size_t i=0; i<v.size(); ++i)
      v[i] = rand01<T>();

    return v;
  }


  template <typename T>
  T max(const std::vector<T>& v) {
    T maximum = v[0];

    for (size_t i=0; i<v.size(); ++i)
      if (v[i] > maximum)
	maximum = v[i];
    return maximum;
  }

  template <typename T>
  void normalize(std::vector<T>& v) {
    T sum = ext::sum(v);
    for (size_t i=0; i<v.size(); ++i)
      v[i] /= sum;
  }

  template <typename T>
  std::vector<size_t> hist(const std::vector<T>& v) {

    T max = ext::max(v);

    std::vector<size_t> h;
    std::map<T, size_t> histogram;
    for (size_t i=0; i<v.size(); ++i) {
      if ( histogram.count(v[i]) == 0)
	histogram[v[i]] = 0;
      ++histogram[v[i]];
    }

    h.resize(max + 1);
    for (size_t i=0; i<h.size(); ++i)
      h[i] = 0;

    typename std::map<T, size_t>::iterator it = histogram.begin();
    for (; it != histogram.end(); ++it)
      h[it->first] = it->second;
    
    return h;
  }

  // ===========================
  // ===== Random Sampling =====
  // ===========================
  template <typename T>
  std::vector<size_t> sampleDataFrom(const std::vector<T>& pdf, size_t nSample) {
    std::vector<size_t> sampledData(nSample);

    std::map<T, size_t> cdf;

    T cumulation = 0;
    for (size_t i=0; i<pdf.size(); ++i)
      cdf[cumulation += pdf[i]] = i;

    for (size_t i=0; i<sampledData.size(); ++i) {
      float linear = rand01<float>();
      sampledData[i] = cdf.upper_bound(linear)->second;
    }

    return sampledData;
  }

  // ===================
  // ===== SoftMax =====
  // ===================
  template <typename T>
  std::vector<T> softmax(const std::vector<T>& x) {
    std::vector<T> s(x.size());

    for (size_t i=0; i<s.size(); ++i)
      s[i] = exp(x[i]);

    T denominator = 1.0 / ext::sum(s);
    for (size_t i=0; i<s.size(); ++i)
      s[i] *= denominator;

    return s;
  }

  // ============================
  // ===== Sigmoid Function =====
  // ============================
  template <typename T>
  std::vector<T> sigmoid(const std::vector<T>& x) {
    std::vector<T> s(x.size());
    std::transform(x.begin(), x.end(), s.begin(), func::sigmoid<T>());
    return s;
  }

  // ================================
  // ===== Biased after Sigmoid =====
  // ================================
  template <typename T>
  std::vector<T> b_sigmoid(const std::vector<T>& x) {
    std::vector<T> s(x.size() + 1);
    std::transform(x.begin(), x.end(), s.begin(), func::sigmoid<T>());
    s.back() = 1.0;
    return s;
  }

  namespace randomgenerator {
    inline time_t srander() {
      time_t t = time(NULL);
      srand(/*t*/0);
      return t;
    }

    static time_t _random_seed_ = srander();
  };
};

#endif
