//
// Created by 付聪 on 2017/6/21.
//

#ifndef EFANNA2E_DISTANCE_H
#define EFANNA2E_DISTANCE_H

#define _USE_MATH_DEFINES
#include <math.h>
#include <x86intrin.h>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <ann/utils/geometry.h>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include "types.h"

namespace efanna2e {
enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
class Distance {
 public:
  virtual float compare(const float* a, const float* b,
                        unsigned length) const = 0;
  virtual ~Distance() {}
};

class DistanceL2 : public Distance {
 public:
  float compare(const float* a, const float* b, unsigned size) const {
    float result = 0;

#ifdef __GNUC__
#ifdef __AVX__

#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                  \
  tmp2 = _mm256_loadu_ps(addr2);                  \
  tmp1 = _mm256_sub_ps(tmp1, tmp2);               \
  tmp1 = _mm256_mul_ps(tmp1, tmp1);               \
  dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (size + 7) & ~7U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float* l = a;
    const float* r = b;
    const float* e_l = l + DD;
    const float* e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR) {
      AVX_L2SQR(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      AVX_L2SQR(l, r, sum, l0, r0);
      AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
             unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm_load_ps(addr1);                      \
  tmp2 = _mm_load_ps(addr2);                      \
  tmp1 = _mm_sub_ps(tmp1, tmp2);                  \
  tmp1 = _mm_mul_ps(tmp1, tmp1);                  \
  dest = _mm_add_ps(dest, tmp1);

    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (size + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float* l = a;
    const float* r = b;
    const float* e_l = l + DD;
    const float* e_r = r + DD;
    float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR) {
      case 12:
        SSE_L2SQR(e_l + 8, e_r + 8, sum, l2, r2);
      case 8:
        SSE_L2SQR(e_l + 4, e_r + 4, sum, l1, r1);
      case 4:
        SSE_L2SQR(e_l, e_r, sum, l0, r0);
      default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      SSE_L2SQR(l, r, sum, l0, r0);
      SSE_L2SQR(l + 4, r + 4, sum, l1, r1);
      SSE_L2SQR(l + 8, r + 8, sum, l2, r2);
      SSE_L2SQR(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];

// normal distance
#else

    float diff0, diff1, diff2, diff3;
    const float* last = a + size;
    const float* unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group) {
      diff0 = a[0] - b[0];
      diff1 = a[1] - b[1];
      diff2 = a[2] - b[2];
      diff3 = a[3] - b[3];
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
      b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last) {
      diff0 = *a++ - *b++;
      result += diff0 * diff0;
    }
#endif
#endif
#endif

    return result;
  }
};

class DistanceInnerProduct : public Distance {
 public:
  float compare(const float* a, const float* b, unsigned size) const {
    float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                \
  tmp2 = _mm256_loadu_ps(addr2);                \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);             \
  dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (size + 7) & ~7U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float* l = a;
    const float* r = b;
    const float* e_l = l + DD;
    const float* e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR) {
      AVX_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      AVX_DOT(l, r, sum, l0, r0);
      AVX_DOT(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
             unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm128_loadu_ps(addr1);                \
  tmp2 = _mm128_loadu_ps(addr2);                \
  tmp1 = _mm128_mul_ps(tmp1, tmp2);             \
  dest = _mm128_add_ps(dest, tmp1);
    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (size + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float* l = a;
    const float* r = b;
    const float* e_l = l + DD;
    const float* e_r = r + DD;
    float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR) {
      case 12:
        SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
      case 8:
        SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
      case 4:
        SSE_DOT(e_l, e_r, sum, l0, r0);
      default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      SSE_DOT(l, r, sum, l0, r0);
      SSE_DOT(l + 4, r + 4, sum, l1, r1);
      SSE_DOT(l + 8, r + 8, sum, l2, r2);
      SSE_DOT(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

    float dot0, dot1, dot2, dot3;
    const float* last = a + size;
    const float* unroll_group = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < unroll_group) {
      dot0 = a[0] * b[0];
      dot1 = a[1] * b[1];
      dot2 = a[2] * b[2];
      dot3 = a[3] * b[3];
      result += dot0 + dot1 + dot2 + dot3;
      a += 4;
      b += 4;
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last) {
      result += *a++ * *b++;
    }
#endif
#endif
#endif
    return result;
  }
};

}

namespace ann {

template <class _T>
struct L2Distance {
  using T = _T;

  static constexpr bool dissimilarity = true;
  static constexpr bool needs_norms = false;

  static inline double get_norm(Tvec_point<T>& pt) {
    if constexpr (std::is_same<T, float>::value) {
      efanna2e::DistanceInnerProduct dist_f;
      float dist = dist_f.compare(pt.coordinates.begin(), pt.coordinates.begin(), pt.coordinates.size());
      return sqrt(dist);
    } else {
      exit(-1);
      return 1.0;
    }
  }

  template <class C, class GetScore>
  static inline void order_candidates(parlay::sequence<C>& c_seq, GetScore get_score) {
    auto less = [&](C a, C b) { return get_score(a) < get_score(b); };
    parlay::sort_inplace(c_seq, less);
  }

  static inline bool compare(float d1, float d2) {
    return d1 < d2;
  }

  static inline bool compare_pair(std::pair<double, int> d1, std::pair<double, int> d2) {
    return d1.first < d2.first;
  }

  static inline float distance(Tvec_point<float>& q, Tvec_point<uint8_t>& p, unsigned d) {
    float result = 0;
    auto q_v = q.coordinates.begin();
    auto p_v = p.coordinates.begin();
    for (int i = 0; i < d; i++) {
      result += (q_v[i] - (float)p_v[i]) * (q_v[i] - (float)p_v[i]);
    }
    return result;
  }

  static inline float distance(Tvec_point<T>& p, Tvec_point<T>& q, unsigned d) {
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
      float result = 0;
      auto q_v = q.coordinates.begin();
      auto p_v = p.coordinates.begin();
      for (int i = 0; i < d; i++) {
        result += ((int32_t)((int16_t)q_v[i] - (int16_t)p_v[i])) *
                  ((int32_t)((int16_t)q_v[i] - (int16_t)p_v[i]));
      }
      return result;
    } else {
      efanna2e::DistanceL2 distfunc;
      return distfunc.compare(p.coordinates.begin(), q.coordinates.begin(), d);
    }
  }
};

template <class _T>
struct AngularDistance {

  using T = _T;

  static inline bool compare_pair(std::pair<double, int> d1, std::pair<double, int> d2) {
//    return d1.first > d2.first;
    return d1.first < d2.first;
  }

  static inline float distance(Tvec_point<float>& q, Tvec_point<uint8_t>& p, unsigned d) {
    exit(-1);
    return 1.0;
  }

  static inline float distance(Tvec_point<T>& p, Tvec_point<T>& q, unsigned d) {
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
      exit(-1);
      return 1.0;
    } else {

//      efanna2e::DistanceInnerProduct distfunc;
//      double norm_p = L2Distance<T>::get_norm(p);
//      double norm_q = L2Distance<T>::get_norm(q);
//
//      double dist = distfunc.compare(p.coordinates.begin(), q.coordinates.begin(), d) / (norm_p * norm_q);
//      return dist;

//      // Inner product of the two vectors
//      efanna2e::DistanceInnerProduct distfunc;
//      double dist = distfunc.compare(p.coordinates.begin(), q.coordinates.begin(), d);
//      // normalize by norms to get cosine similarity
//      double cosine_sim = dist / (static_cast<double>(p.norm) * static_cast<double>(q.norm));
//      return 1 - cosine_sim;

//      efanna2e::DistanceInnerProduct distfunc;
//      return distfunc.compare(p.coordinates.begin(), q.coordinates.begin(), d);

      // Inner product of the two vectors
      efanna2e::DistanceInnerProduct distfunc;
      double cosine_sim = distfunc.compare(p.coordinates.begin(), q.coordinates.begin(), d);
      double norm_p = L2Distance<T>::get_norm(p);
      double norm_q = L2Distance<T>::get_norm(q);
      cosine_sim /= (norm_p * norm_q);

      return acos(cosine_sim);


//      double angular_dist = (static_cast<double>(1) / cosine_sim) * (1 / M_PI);
//      return angular_dist;
    }
  }
};

}  // namespace ann


#endif  // EFANNA2E_DISTANCE_H