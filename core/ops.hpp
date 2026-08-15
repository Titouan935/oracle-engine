#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────
// Tenseur 2D simple (rangée principale)
// ─────────────────────────────────────────────────────────────────────────
struct Tensor {
    std::vector<float> data;
    int rows = 0;
    int cols = 0;

    Tensor() = default;
    Tensor(int r, int c) : data(r * c, 0.f), rows(r), cols(c) {}

    void resize(int r, int c) {
        rows = r; cols = c;
        data.assign(r * c, 0.f);
    }
    void zero() { std::fill(data.begin(), data.end(), 0.f); }

    float& at(int r, int c)       { return data[r * cols + c]; }
    float  at(int r, int c) const { return data[r * cols + c]; }

    float*       ptr(int r = 0)       { return data.data() + r * cols; }
    const float* ptr(int r = 0) const { return data.data() + r * cols; }

    int n_elems() const { return rows * cols; }
};

// ─────────────────────────────────────────────────────────────────────────
// Opérations mathématiques du transformer
// ─────────────────────────────────────────────────────────────────────────
namespace ops {

// ── Normalisation ─────────────────────────────────────────────────────────

// RMS Normalization : out[i] = x[i] / sqrt(mean(x²) + eps) * weight[i]
void rmsnorm(float* out, const float* x, const float* weight, int n,
             float eps = 1e-5f);

// ── Multiplication matricielle ────────────────────────────────────────────

// out = A × B^T    (A: m×k,  B: n×k,  out: m×n)
// Utilisé pour projections : out = x × W^T
void matmul(float* out, const float* A, const float* B,
            int m, int n, int k);

// Variante : out = A × B   (A: m×k,  B: k×n,  out: m×n)
void matmul_nn(float* out, const float* A, const float* B,
               int m, int n, int k);

// ── Activation ───────────────────────────────────────────────────────────

// SiLU (Swish) en place : x[i] = x[i] * sigmoid(x[i])
void silu(float* x, int n);

// ── Opérations élémentaires ───────────────────────────────────────────────

// Multiplication élément par élément : out[i] = a[i] * b[i]
void mul(float* out, const float* a, const float* b, int n);

// Addition en place : a[i] += b[i]
void add(float* a, const float* b, int n);

// Copie : dst[i] = src[i]
void copy(float* dst, const float* src, int n);

// ── Distribution ─────────────────────────────────────────────────────────

// Softmax en place : x[i] = exp(x[i] - max) / sum(exp)
void softmax(float* x, int n);

// ── Encodage de position rotatif (RoPE) ──────────────────────────────────
//
// Applique RoPE aux vecteurs q et k pour la position `pos`.
// q, k  : [n_heads × head_dim] et [n_kv_heads × head_dim]
// Modifie q et k en place.
void rope(float* q, float* k,
          int n_heads, int n_kv_heads, int head_dim,
          int pos, float freq_base = 500000.f);

// ── Utilitaires ──────────────────────────────────────────────────────────

// Retourne l'index du maximum
int argmax(const float* x, int n);

// Produit scalaire : sum(a[i] * b[i])
float dot(const float* a, const float* b, int n);

} // namespace ops
