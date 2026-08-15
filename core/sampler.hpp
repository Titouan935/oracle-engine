#pragma once
#include <vector>
#include <cstdint>
#include <random>

// ─────────────────────────────────────────────────────────────────────────
// Sampler — choisit le prochain token depuis les logits
//
// Paramètres :
//   temperature  : diversité (0 = déterministe, 1 = normal, >1 = créatif)
//   top_p        : nucleus sampling (0.9 = garde 90% de la masse prob.)
//   top_k        : limite aux top_k tokens les plus probables (0 = désactivé)
//   repeat_pen   : pénalité de répétition (1.0 = aucune, 1.1 = légère)
//   repeat_last  : fenêtre de répétition (en tokens)
// ─────────────────────────────────────────────────────────────────────────
struct SamplerParams {
    float    temperature  = 0.7f;
    float    top_p        = 0.9f;
    int      top_k        = 40;
    float    repeat_pen   = 1.1f;
    int      repeat_last  = 64;
};

class Sampler {
public:
    explicit Sampler(const SamplerParams& p = {}, uint64_t seed = 42);

    // Choisit un token depuis les logits [n_vocab]
    // Modifie logits en place (normalisation, pénalités)
    int32_t sample(float* logits, int n_vocab);

    // Enregistre un token généré (pour la pénalité de répétition)
    void record(int32_t token_id);

    // Réinitialise l'historique
    void reset() { history_.clear(); }

    void set_params(const SamplerParams& p) { p_ = p; }

private:
    SamplerParams           p_;
    std::mt19937            rng_;
    std::vector<int32_t>    history_;

    void apply_repeat_penalty(float* logits, int n_vocab);
    void apply_temperature(float* logits, int n_vocab);
    int32_t sample_top_pk(float* logits, int n_vocab);
};
