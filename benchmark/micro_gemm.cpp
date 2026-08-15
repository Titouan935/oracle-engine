// ─────────────────────────────────────────────────────────────────────────
// micro_gemm.cpp — Micro-benchmark ISOLÉ du kernel gemm_q (prefill batché)
//
// Teste gemm_q sur UN seul tenseur du modèle → empreinte RAM minime (seules
// les pages de ce tenseur sont touchées), sans le bruit du paging / threading
// d'un run complet. Mesure deux choses :
//
//   1. CORRECTION — compare la sortie de gemm_q à une référence scalaire
//      (dequantize_row + produit scalaire naïf). Doit matcher à ~1e-4.
//   2. VITESSE — compare l'ANCIEN schéma (dequant + ops::dot par token) au
//      NOUVEAU kernel tuilé (gemm_q), en GFLOP/s + ratio d'accélération.
//
// Usage : micro_gemm -m models/x.gguf [-t threads] [-T tokens] [-r iters]
// ─────────────────────────────────────────────────────────────────────────
#include "core/gguf_parser.hpp"
#include "core/ops.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using clk = std::chrono::high_resolution_clock;

// Ancien schéma gemm_q (avant tuilage) : dequant une fois, ops::dot par token.
static void gemm_ref_pertoken(const GGUFParser& g, const std::string& name,
                              float* out, const float* x,
                              int n_tokens, int n_out, int n_in) {
    #pragma omp parallel
    {
        std::vector<float> row(n_in);
        #pragma omp for schedule(static)
        for (int j = 0; j < n_out; j++) {
            g.dequantize_row(name, j, row.data());
            for (int t = 0; t < n_tokens; t++)
                out[(size_t)t * n_out + j] = ops::dot(x + (size_t)t * n_in, row.data(), n_in);
        }
    }
}

int main(int argc, char** argv) {
    std::string model;
    int threads = 0, T = 32, iters = 10;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if      (k == "-m" && i+1 < argc) model   = argv[++i];
        else if (k == "-t" && i+1 < argc) threads = std::atoi(argv[++i]);
        else if (k == "-T" && i+1 < argc) T       = std::atoi(argv[++i]);
        else if (k == "-r" && i+1 < argc) iters   = std::atoi(argv[++i]);
    }
    if (model.empty()) { std::fprintf(stderr, "usage: micro_gemm -m <model.gguf> [-t N -T N -r N]\n"); return 2; }
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    int th = omp_get_max_threads();
#else
    int th = 1;
#endif

    GGUFParser g;
    if (!g.load(model)) { std::fprintf(stderr, "load échoué: %s\n", model.c_str()); return 1; }

    // Choisit le plus gros tenseur 2D Q4_K avec n_in multiple de 256.
    std::string best; uint64_t best_sz = 0; int n_in = 0, n_out = 0;
    for (const auto& nm : g.tensor_names()) {
        const TensorInfo& ti = g.tensor_info(nm);
        if (ti.type != GGMLType::Q4_K || ti.shape.size() != 2) continue;
        if (ti.shape[0] % 256 != 0) continue;
        uint64_t sz = ti.shape[0] * ti.shape[1];
        if (sz > best_sz) { best_sz = sz; best = nm; n_in = (int)ti.shape[0]; n_out = (int)ti.shape[1]; }
    }
    if (best.empty()) { std::fprintf(stderr, "aucun tenseur Q4_K 2D trouvé\n"); return 1; }

    std::printf("═══════════════════════════════════════════════════════════\n");
    std::printf("  micro_gemm — kernel gemm_q isolé\n");
    std::printf("═══════════════════════════════════════════════════════════\n");
    std::printf("Tenseur : %s  [n_out=%d × n_in=%d, Q4_K]\n", best.c_str(), n_out, n_in);
    std::printf("Tokens  : %d | Threads : %d | Iters : %d\n\n", T, th, iters);

    // Activations aléatoires déterministes.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> x((size_t)T * n_in);
    for (auto& v : x) v = U(rng);

    std::vector<float> out_new((size_t)T * n_out, 0.f);
    std::vector<float> out_old((size_t)T * n_out, 0.f);
    std::vector<float> out_ref((size_t)T * n_out, 0.f);

    // ── Référence scalaire (correction) ──────────────────────────────────
    {
        std::vector<float> row(n_in);
        for (int j = 0; j < n_out; j++) {
            g.dequantize_row(best, j, row.data());
            for (int t = 0; t < T; t++) {
                double s = 0.0;
                const float* xt = x.data() + (size_t)t * n_in;
                for (int k = 0; k < n_in; k++) s += (double)xt[k] * row[k];
                out_ref[(size_t)t * n_out + j] = (float)s;
            }
        }
    }

    // ── gemm_q (nouveau, tuilé) ──────────────────────────────────────────
    if (!g.gemm_q(best, out_new.data(), x.data(), T, n_out, n_in)) {
        std::fprintf(stderr, "gemm_q a retourné false\n"); return 1;
    }
    // ── ancien schéma per-token ──────────────────────────────────────────
    gemm_ref_pertoken(g, best, out_old.data(), x.data(), T, n_out, n_in);

    // Correction : max diff relatif new vs référence.
    double max_rel = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < out_new.size(); i++) {
        double d = std::fabs((double)out_new[i] - out_ref[i]);
        double r = d / (std::fabs((double)out_ref[i]) + 1e-6);
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
    }
    std::printf("Correction (gemm_q tuilé vs référence scalaire) :\n");
    std::printf("  max abs diff = %.3e | max rel diff = %.3e  →  %s\n\n",
                max_abs, max_rel, (max_rel < 1e-3 ? "OK ✓" : "ÉCART SUSPECT ✗"));

    // ── Vitesse ──────────────────────────────────────────────────────────
    const double gflop = 2.0 * (double)n_out * n_in * T / 1e9;   // par appel

    // warmup
    g.gemm_q(best, out_new.data(), x.data(), T, n_out, n_in);
    auto t0 = clk::now();
    for (int r = 0; r < iters; r++) g.gemm_q(best, out_new.data(), x.data(), T, n_out, n_in);
    double dt_new = std::chrono::duration<double>(clk::now() - t0).count();

    gemm_ref_pertoken(g, best, out_old.data(), x.data(), T, n_out, n_in);
    t0 = clk::now();
    for (int r = 0; r < iters; r++) gemm_ref_pertoken(g, best, out_old.data(), x.data(), T, n_out, n_in);
    double dt_old = std::chrono::duration<double>(clk::now() - t0).count();

    double g_new = iters * gflop / dt_new;
    double g_old = iters * gflop / dt_old;

    std::printf("Vitesse (médiane implicite sur %d itérations) :\n", iters);
    std::printf("  Ancien (dequant + ops::dot/token) : %6.2f GFLOP/s  (%.3fs)\n", g_old, dt_old);
    std::printf("  Nouveau (kernel tuilé 4×2)        : %6.2f GFLOP/s  (%.3fs)\n", g_new, dt_new);
    std::printf("  Accélération                       : %.2f×\n", g_new / g_old);
    std::printf("═══════════════════════════════════════════════════════════\n");
    return 0;
}
