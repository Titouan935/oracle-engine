#include "ops.hpp"
#include <cmath>
#include <cassert>
#include <limits>
#include <numeric>
#include <cstring>
#include <algorithm>

// ── SIMD ──────────────────────────────────────────────────────────────────
#ifdef __AVX2__
#include <immintrin.h>

static inline float hsum_avx(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// exp(x) vectorisé — approximation Cephes (avx_mathfun), erreur relative ~1e-6.
// Utilisée par silu() et softmax() pour éviter la boucle std::exp scalaire.
static inline __m256 exp256_ps(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.f);
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

    // fx = floor(x * log2(e) + 0.5)
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f),
                                _mm256_set1_ps(0.5f));
    __m256 tmp  = _mm256_floor_ps(fx);
    __m256 mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
    fx = _mm256_sub_ps(tmp, mask);

    // x -= fx * ln2  (ln2 scindé en deux constantes pour la précision)
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f),    x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 z = _mm256_mul_ps(x, x);

    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    // y *= 2^fx  (reconstruction de l'exposant)
    __m256i imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0 = _mm256_slli_epi32(imm0, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(imm0));
}

#elif defined(__ARM_NEON)
#include <arm_neon.h>

static inline float hsum_neon(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

// exp(x) vectorisé NEON — même approximation Cephes (neon_mathfun).
static inline float32x4_t exp_ps_neon(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.f);
    x = vminq_f32(x, vdupq_n_f32( 88.3762626647949f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));

    float32x4_t fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f));
    float32x4_t tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));   // trunc
    uint32x4_t mask = vandq_u32(vcgtq_f32(tmp, fx), vreinterpretq_u32_f32(one));
    fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));     // floor

    x = vmlsq_f32(x, fx, vdupq_n_f32(0.693359375f));
    x = vmlsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
    y = vmlaq_f32(vdupq_n_f32(1.3981999507E-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(8.3334519073E-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(4.1665795894E-2f), y, x);
    y = vmlaq_f32(vdupq_n_f32(1.6666665459E-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(5.0000001201E-1f), y, x);
    y = vmlaq_f32(x, y, z);
    y = vaddq_f32(y, one);

    int32x4_t emm = vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(0x7f));
    emm = vshlq_n_s32(emm, 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(emm));
}
#endif

namespace ops {

// ══════════════════════════════════════════════════════════════════════════
//  Produit scalaire — AVX2 FMA
//  Accélération ×8 sur le chemin critique (attention + LM head)
// ══════════════════════════════════════════════════════════════════════════
float dot(const float* a, const float* b, int n) {
#ifdef __AVX2__
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    float s = hsum_avx(_mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                      _mm256_add_ps(acc2, acc3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#elif defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f);
    float32x4_t acc3 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i),    vld1q_f32(b+i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    for (; i + 4 <= n; i += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a+i), vld1q_f32(b+i));
    float s = hsum_neon(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  RMS Normalization — AVX2
//  out[i] = (x[i] / rms(x)) * weight[i]
// ══════════════════════════════════════════════════════════════════════════
void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = hsum_avx(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    ss = 1.f / std::sqrt(ss / (float)n + eps);

    __m256 vss = _mm256_set1_ps(ss);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vss), vw));
    }
    for (; i < n; i++) out[i] = x[i] * ss * weight[i];
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float ss = hsum_neon(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    ss = 1.f / std::sqrt(ss / (float)n + eps);

    float32x4_t vss = vdupq_n_f32(ss);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vss), vw));
    }
    for (; i < n; i++) out[i] = x[i] * ss * weight[i];
#else
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss /= (float)n;
    ss  = 1.f / std::sqrt(ss + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * weight[i];
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  Multiplication matricielle — A × B^T    (AVX2 FMA, déroulé ×4 en j)
//  A   : m × k
//  B   : n × k  (utilisée transposée → n lignes de k colonnes)
//  out : m × n
//
//  Cas principal : m = 1 (génération token à token)
//  On calcule n produits scalaires de longueur k.
//  Dérouler j ×4 : réutilise le vecteur A chargé pour 4 sorties simultanées.
// ══════════════════════════════════════════════════════════════════════════
void matmul(float* out, const float* A, const float* B, int m, int n, int k) {
#ifdef __AVX2__
    for (int i = 0; i < m; i++) {
        const float* ar = A + i * k;
        float*       or_ = out + i * n;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            const float* b0 = B + (j+0) * k;
            const float* b1 = B + (j+1) * k;
            const float* b2 = B + (j+2) * k;
            const float* b3 = B + (j+3) * k;

            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

            int l = 0;
            for (; l + 8 <= k; l += 8) {
                __m256 va = _mm256_loadu_ps(ar + l);
                a0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + l), a0);
                a1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + l), a1);
                a2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + l), a2);
                a3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + l), a3);
            }
            float r0 = hsum_avx(a0), r1 = hsum_avx(a1);
            float r2 = hsum_avx(a2), r3 = hsum_avx(a3);
            for (; l < k; l++) {
                r0 += ar[l] * b0[l]; r1 += ar[l] * b1[l];
                r2 += ar[l] * b2[l]; r3 += ar[l] * b3[l];
            }
            or_[j] = r0; or_[j+1] = r1; or_[j+2] = r2; or_[j+3] = r3;
        }
        for (; j < n; j++) {
            or_[j] = dot(ar, B + j * k, k);
        }
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < m; i++) {
        const float* ar = A + i * k;
        float*       or_ = out + i * n;

        int j = 0;
        for (; j + 4 <= n; j += 4) {
            const float* b0 = B + (j+0) * k;
            const float* b1 = B + (j+1) * k;
            const float* b2 = B + (j+2) * k;
            const float* b3 = B + (j+3) * k;

            float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
            float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);

            int l = 0;
            for (; l + 4 <= k; l += 4) {
                float32x4_t va = vld1q_f32(ar + l);
                a0 = vfmaq_f32(a0, va, vld1q_f32(b0 + l));
                a1 = vfmaq_f32(a1, va, vld1q_f32(b1 + l));
                a2 = vfmaq_f32(a2, va, vld1q_f32(b2 + l));
                a3 = vfmaq_f32(a3, va, vld1q_f32(b3 + l));
            }
            float r0 = hsum_neon(a0), r1 = hsum_neon(a1);
            float r2 = hsum_neon(a2), r3 = hsum_neon(a3);
            for (; l < k; l++) {
                r0 += ar[l] * b0[l]; r1 += ar[l] * b1[l];
                r2 += ar[l] * b2[l]; r3 += ar[l] * b3[l];
            }
            or_[j] = r0; or_[j+1] = r1; or_[j+2] = r2; or_[j+3] = r3;
        }
        for (; j < n; j++) {
            or_[j] = dot(ar, B + j * k, k);
        }
    }
#else
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            const float* a_row = A + i * k;
            const float* b_row = B + j * k;
            for (int l = 0; l < k; l++) acc += a_row[l] * b_row[l];
            out[i * n + j] = acc;
        }
    }
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  Multiplication matricielle — A × B  (sans transposition)
// ══════════════════════════════════════════════════════════════════════════
void matmul_nn(float* out, const float* A, const float* B, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.f;
            for (int l = 0; l < k; l++)
                acc += A[i * k + l] * B[l * n + j];
            out[i * n + j] = acc;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  SiLU  (Sigmoid Linear Unit / Swish)
// ══════════════════════════════════════════════════════════════════════════
void silu(float* x, int n) {
#ifdef __AVX2__
    const __m256 one  = _mm256_set1_ps(1.f);
    const __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        // x * sigmoid(x) = x / (1 + exp(-x))
        __m256 e = exp256_ps(_mm256_sub_ps(zero, v));
        _mm256_storeu_ps(x + i, _mm256_div_ps(v, _mm256_add_ps(one, e)));
    }
    for (; i < n; i++) x[i] = x[i] / (1.f + std::exp(-x[i]));
#elif defined(__ARM_NEON)
    const float32x4_t one  = vdupq_n_f32(1.f);
    const float32x4_t zero = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t e = exp_ps_neon(vsubq_f32(zero, v));
        vst1q_f32(x + i, vdivq_f32(v, vaddq_f32(one, e)));
    }
    for (; i < n; i++) x[i] = x[i] / (1.f + std::exp(-x[i]));
#else
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.f + std::exp(-x[i]));
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  Multiplication élément par élément — AVX2
// ══════════════════════════════════════════════════════════════════════════
void mul(float* out, const float* a, const float* b, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) out[i] = a[i] * b[i];
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  Addition en place — AVX2
// ══════════════════════════════════════════════════════════════════════════
void add(float* a, const float* b, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(a + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) a[i] += b[i];
#elif defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < n; i++) a[i] += b[i];
#else
    for (int i = 0; i < n; i++) a[i] += b[i];
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  Copie
// ══════════════════════════════════════════════════════════════════════════
void copy(float* dst, const float* src, int n) {
    std::memcpy(dst, src, n * sizeof(float));
}

// ══════════════════════════════════════════════════════════════════════════
//  Softmax en place (numériquement stable)
// ══════════════════════════════════════════════════════════════════════════
void softmax(float* x, int n) {
    if (n <= 0) return;
    float max_val = *std::max_element(x, x + n);
    float sum = 0.f;
#ifdef __AVX2__
    {
        const __m256 vmax = _mm256_set1_ps(max_val);
        __m256 vsum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 e = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), vmax));
            _mm256_storeu_ps(x + i, e);
            vsum = _mm256_add_ps(vsum, e);
        }
        sum = hsum_avx(vsum);
        for (; i < n; i++) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    }
#elif defined(__ARM_NEON)
    {
        const float32x4_t vmax = vdupq_n_f32(max_val);
        float32x4_t vsum = vdupq_n_f32(0.f);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t e = exp_ps_neon(vsubq_f32(vld1q_f32(x + i), vmax));
            vst1q_f32(x + i, e);
            vsum = vaddq_f32(vsum, e);
        }
        sum = hsum_neon(vsum);
        for (; i < n; i++) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    }
#else
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
#endif
    float inv = 1.f / sum;
#ifdef __AVX2__
    __m256 vinv = _mm256_set1_ps(inv);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    for (; i < n; i++) x[i] *= inv;
#elif defined(__ARM_NEON)
    float32x4_t vinv = vdupq_n_f32(inv);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vinv));
    for (; i < n; i++) x[i] *= inv;
#else
    for (int i = 0; i < n; i++) x[i] *= inv;
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  RoPE — Rotary Position Embedding
// ══════════════════════════════════════════════════════════════════════════
void rope(float* q, float* k,
          int n_heads, int n_kv_heads, int head_dim,
          int pos, float freq_base) {
    const int half = head_dim / 2;

    // Optimisation : cos/sin (et le std::pow) ne dépendent que de (pos, i), pas
    // de la tête. On les calcule UNE fois par position et on les réutilise pour
    // toutes les têtes → ~n_heads fois moins d'appels transcendantaux, résultat
    // strictement identique.
    // Buffer de pile pour le cas courant (half <= 128, soit head_dim <= 256) ;
    // repli sur le tas si un modèle a un head_dim plus grand (pas de débordement).
    float stack_cos[128], stack_sin[128];
    std::vector<float> heap_cos, heap_sin;
    float* cos_t = stack_cos;
    float* sin_t = stack_sin;
    if (half > 128) {
        heap_cos.resize(half);
        heap_sin.resize(half);
        cos_t = heap_cos.data();
        sin_t = heap_sin.data();
    }
    for (int i = 0; i < half; i++) {
        float theta = (float)pos
                    / std::pow(freq_base, (float)(2 * i) / (float)head_dim);
        cos_t[i] = std::cos(theta);
        sin_t[i] = std::sin(theta);
    }

    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float q0 = qh[i], q1 = qh[i + half];
            qh[i]        = q0 * cos_t[i] - q1 * sin_t[i];
            qh[i + half] = q0 * sin_t[i] + q1 * cos_t[i];
        }
    }

    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float k0 = kh[i], k1 = kh[i + half];
            kh[i]        = k0 * cos_t[i] - k1 * sin_t[i];
            kh[i + half] = k0 * sin_t[i] + k1 * cos_t[i];
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  Argmax
// ══════════════════════════════════════════════════════════════════════════
int argmax(const float* x, int n) {
    if (n <= 0) return 0;
    int   best = 0;
    float val  = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > val) { val = x[i]; best = i; }
    return best;
}

} // namespace ops
