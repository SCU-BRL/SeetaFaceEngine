/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Detection module, containing codes implementing the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#ifndef SEETA_FD_UTIL_MATH_FUNC_H_
#define SEETA_FD_UTIL_MATH_FUNC_H_

#ifdef USE_SSE
#include <immintrin.h>
#elif USE_ARM_NEON
#include <arm_neon.h>
#endif

#include <cstdint>

namespace seeta {
namespace fd {

class MathFunction {
 public:
  static inline void UInt8ToInt32(const uint8_t* src, int32_t* dest,
      int32_t len) {
    for (int32_t i = 0; i < len; i++)
      *(dest++) = static_cast<int32_t>(*(src++));
  }

  static inline void VectorAdd(const int32_t* x, const int32_t* y, int32_t* z,
      int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i x1;
    __m128i y1;
    const __m128i* x2 = reinterpret_cast<const __m128i*>(x);
    const __m128i* y2 = reinterpret_cast<const __m128i*>(y);
    __m128i* z2 = reinterpret_cast<__m128i*>(z);

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_si128(x2++);
      y1 = _mm_loadu_si128(y2++);
      _mm_storeu_si128(z2++, _mm_add_epi32(x1, y1));
    }
    for (; i < len; i++)
      *(z + i) = (*(x + i)) + (*(y + i));
#elif USE_ARM_NEON
    int32x4_t x1;
    int32x4_t y1;
    const int32_t *px = reinterpret_cast<const int32_t*>(x);
    const int32_t *py = reinterpret_cast<const int32_t*>(y);
    int32_t *pz = reinterpret_cast<int32_t*>(z);

    for(i=0;i<len-4;i +=4)
    {
        x1 = vld1q_s32(px+i);
        y1 = vld1q_s32(py+i);
        vst1q_s32(pz+i,vaddq_s32(x1,y1));
    }
    for(;i< len;i++)
        *(z+i) = (*(x+i)) + (*(y+i));
#else
    for (i = 0; i < len; i++)
      *(z + i) = (*(x + i)) + (*(y + i));
#endif
  }

  static inline void VectorSub(const int32_t* x, const int32_t* y, int32_t* z,
      int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i x1;
    __m128i y1;
    const __m128i* x2 = reinterpret_cast<const __m128i*>(x);
    const __m128i* y2 = reinterpret_cast<const __m128i*>(y);
    __m128i* z2 = reinterpret_cast<__m128i*>(z);

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_si128(x2++);
      y1 = _mm_loadu_si128(y2++);

      _mm_storeu_si128(z2++, _mm_sub_epi32(x1, y1));
    }
    for (; i < len; i++)
      *(z + i) = (*(x + i)) - (*(y + i));
#elif USE_ARM_NEON
    int32x4_t x1;
    int32x4_t y1;
    const int32_t *px = reinterpret_cast<const int32_t*>(x);
    const int32_t *py = reinterpret_cast<const int32_t*>(y);
    int32_t *pz = reinterpret_cast<int32_t*>(z);

    for(i=0;i<len-4;i +=4)
    {
        x1 = vld1q_s32(px+i);
        y1 = vld1q_s32(py+i);
        vst1q_s32(pz+i,vsubq_s32(x1,y1));
    }
    for(;i< len;i++)
        *(z+i) = (*(x+i)) - (*(y+i));
#else
    for (i = 0; i < len; i++)
      *(z + i) = (*(x + i)) - (*(y + i));
#endif
  }

  static inline void VectorAbs(const int32_t* src, int32_t* dest, int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i val;
    __m128i val_abs;
    const __m128i* x = reinterpret_cast<const __m128i*>(src);
    __m128i* y = reinterpret_cast<__m128i*>(dest);

    for (i = 0; i < len - 4; i += 4) {
      val = _mm_loadu_si128(x++);
      val_abs = _mm_abs_epi32(val);
      _mm_storeu_si128(y++, val_abs);
    }
    for (; i < len; i++)
      dest[i] = (src[i] >= 0 ? src[i] : -src[i]);
#elif USE_ARM_NEON
    int32x4_t val;
    int32x4_t val_abs;
    const int32_t *px = reinterpret_cast<const int32_t*>(src);
    int32_t *pz = reinterpret_cast<int32_t*>(dest);

    for(i=0;i<len-4;i +=4)
    {
        val = vld1q_s32(px+i);
        val_abs = vabsq_s32(val);
        vst1q_s32(pz+i,val_abs);
    }
    for(;i< len;i++)
        *(dest+i) = (*(src+i)) > 0 ? (*(src+i)) : -(*(src+i));
#else
    for (i = 0; i < len; i++)
      dest[i] = (src[i] >= 0 ? src[i] : -src[i]);
#endif
  }

  static inline void Square(const int32_t* src, uint32_t* dest, int32_t len) {
    int32_t i;
#ifdef USE_SSE
    __m128i x1;
    const __m128i* x2 = reinterpret_cast<const __m128i*>(src);
    __m128i* y2 = reinterpret_cast<__m128i*>(dest);

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_si128(x2++);
      _mm_storeu_si128(y2++, _mm_mullo_epi32(x1, x1));
    }
    for (; i < len; i++)
      *(dest + i) = (*(src + i)) * (*(src + i));
#elif USE_ARM_NEON
    int32x4_t x1;
    const int32_t *px = reinterpret_cast<const int32_t*>(src);
    int32_t *pz = reinterpret_cast<int32_t*>(dest);

    for(i=0;i<len-4;i +=4)
    {
        x1 = vld1q_s32(px+i);
        vst1q_s32(pz+i,vmulq_s32(x1,x1));
    }
    for(;i< len;i++)
        *(dest+i) = (*(src+i)) * (*(src+i));
#else
    for (i = 0; i < len; i++)
      *(dest + i) = (*(src + i)) * (*(src + i));
#endif
  }

  static inline float VectorInnerProduct(const float* x, const float* y,
      int32_t len) {
    float prod = 0;
    int32_t i;
#ifdef USE_SSE
    __m128 x1;
    __m128 y1;
    __m128 z1 = _mm_setzero_ps();
    float buf[4];

    for (i = 0; i < len - 4; i += 4) {
      x1 = _mm_loadu_ps(x + i);
      y1 = _mm_loadu_ps(y + i);
      z1 = _mm_add_ps(z1, _mm_mul_ps(x1, y1));
    }
    _mm_storeu_ps(&buf[0], z1);
    prod = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < len; i++)
      prod += x[i] * y[i];
#elif USE_ARM_NEON
    float32x4_t x1;
    float32x4_t y1;
    float32x4_t z1 = vmovq_n_f32(0.0);
    float prod;
    float buf[4];

    for(i=0;i<len-4;i +=4)
    {
        x1 = vld1q_f32(x+i);
        y1 = vld1q_f32(y+i);
        // z1 = z1 + x1 * y1
        z1 = vmlaq_f32(z1,x1,y1);
    }
    vst1q_f32(&buf[0],z1);
    prod = buf[0] + buf[1] + buf[2] + buf[3];
    for(;i<len;i++)
        prod += x[i] *y[i];
#else
    for (i = 0; i < len; i++)
        prod += x[i] * y[i];
#endif
    return prod;
  }
};

}  // namespace fd
}  // namespace seeta

#endif  // SEETA_FD_UTIL_MATH_FUNC_H_
