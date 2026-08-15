#include "sampler.hpp"
#include "ops.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

Sampler::Sampler(const SamplerParams& p, uint64_t seed)
    : p_(p), rng_(seed) {}

// ══════════════════════════════════════════════════════════════════════════
//  Pénalité de répétition
//  logits[id] /= repeat_pen  si le token est dans l'historique récent
//  (si logit > 0, on divise ; si logit < 0, on multiplie pour amplifier)
// ══════════════════════════════════════════════════════════════════════════
void Sampler::apply_repeat_penalty(float* logits, int n_vocab) {
    if (p_.repeat_pen <= 1.f || history_.empty()) return;

    int start = (int)history_.size() - p_.repeat_last;
    if (start < 0) start = 0;

    for (int i = start; i < (int)history_.size(); i++) {
        int32_t id = history_[i];
        if (id < 0 || id >= n_vocab) continue;
        if (logits[id] > 0.f) logits[id] /= p_.repeat_pen;
        else                   logits[id] *= p_.repeat_pen;
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  Température : divise les logits pour contrôler l'entropie
//  T→0 : distribution piquée (déterministe)
//  T→1 : distribution naturelle du modèle
//  T→∞ : distribution uniforme (aléatoire)
// ══════════════════════════════════════════════════════════════════════════
void Sampler::apply_temperature(float* logits, int n_vocab) {
    if (p_.temperature <= 0.f || p_.temperature == 1.f) return;
    float inv = 1.f / p_.temperature;
    for (int i = 0; i < n_vocab; i++) logits[i] *= inv;
}

// ══════════════════════════════════════════════════════════════════════════
//  Top-K + Top-P (Nucleus Sampling)
//
//  1. Garde les top_k tokens les plus probables (top_k = 0 → tous)
//  2. Parmi eux, garde le noyau dont la somme de probabilité ≥ top_p
//  3. Tire au sort parmi le noyau retenu
// ══════════════════════════════════════════════════════════════════════════
int32_t Sampler::sample_top_pk(float* logits, int n_vocab) {
    // Cas déterministe (T=0)
    if (p_.temperature <= 0.f) return ops::argmax(logits, n_vocab);

    // Convertit logits → probabilités (softmax)
    ops::softmax(logits, n_vocab);

    // Trie les indices par probabilité décroissante
    std::vector<int32_t> idx(n_vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(),
                      idx.begin() + std::min(p_.top_k > 0 ? p_.top_k : n_vocab, n_vocab),
                      idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    // Applique top_k
    int k = (p_.top_k > 0) ? std::min(p_.top_k, n_vocab) : n_vocab;

    // Applique top_p : coupe dès que la masse cumulée ≥ top_p
    float cum = 0.f;
    int cut = k;
    for (int i = 0; i < k; i++) {
        cum += logits[idx[i]];
        if (cum >= p_.top_p) { cut = i + 1; break; }
    }

    // Tire un token dans le noyau [0..cut)
    // (distribution uniforme sur les probabilités normalisées)
    float total = 0.f;
    for (int i = 0; i < cut; i++) total += logits[idx[i]];
    std::uniform_real_distribution<float> uni(0.f, total);
    float r = uni(rng_);

    for (int i = 0; i < cut; i++) {
        r -= logits[idx[i]];
        if (r <= 0.f) return idx[i];
    }
    return idx[0];  // fallback
}

// ══════════════════════════════════════════════════════════════════════════
//  Point d'entrée principal
// ══════════════════════════════════════════════════════════════════════════
int32_t Sampler::sample(float* logits, int n_vocab) {
    apply_repeat_penalty(logits, n_vocab);
    apply_temperature(logits, n_vocab);
    int32_t tok = sample_top_pk(logits, n_vocab);
    record(tok);
    return tok;
}

void Sampler::record(int32_t token_id) {
    history_.push_back(token_id);
    // Garde seulement les repeat_last tokens
    if ((int)history_.size() > p_.repeat_last * 2)
        history_.erase(history_.begin(),
                       history_.begin() + p_.repeat_last);
}
