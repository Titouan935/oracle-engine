#pragma once
#include "gguf_parser.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include <string>
#include <vector>
#include <cstdint>

// ── Configuration du modèle (lue depuis les métadonnées GGUF) ─────────────
struct ModelConfig {
    uint32_t n_vocab    = 0;
    uint32_t n_layers   = 0;
    uint32_t n_heads    = 0;      // têtes de requête (Q)
    uint32_t n_kv_heads = 0;      // têtes clé/valeur (K, V) — GQA
    uint32_t embed_dim  = 0;      // dimension d'embedding
    uint32_t ffn_hidden = 0;      // dimension cachée du FFN
    uint32_t head_dim   = 0;      // dim par tête (lu du GGUF ; ≠ embed_dim/n_heads pour Qwen3)
    uint32_t ctx_len    = 4096;   // longueur de contexte utilisée
    float    rope_freq  = 500000.f;
    float    rms_eps    = 1e-5f;  // epsilon des RMSNorm (lu du GGUF)
    bool     qk_norm    = false;  // Qwen3 : RMSNorm par tête sur Q et K avant RoPE

    // Calculé
    uint32_t n_gqa()     const { return n_heads / n_kv_heads; }   // têtes par groupe KV
    uint32_t n_q_total() const { return n_heads * head_dim; }
    uint32_t n_kv_total()const { return n_kv_heads * head_dim; }
};

// ── Cache clé-valeur (KV Cache) ───────────────────────────────────────────
// Stocke les K et V passés pour éviter de les recalculer.
// Layout : [n_layers][ctx_len][n_kv_heads × head_dim]
struct KVCache {
    std::vector<float> k;  // clés
    std::vector<float> v;  // valeurs

    void init(const ModelConfig& cfg) {
        size_t n = (size_t)cfg.n_layers * cfg.ctx_len * cfg.n_kv_total();
        k.assign(n, 0.f);
        v.assign(n, 0.f);
    }

    // Pointeur vers K[layer][pos][:]
    float* kptr(int layer, int pos, const ModelConfig& cfg) {
        return k.data()
            + (size_t)layer * cfg.ctx_len * cfg.n_kv_total()
            + (size_t)pos   * cfg.n_kv_total();
    }
    // Pointeur vers V[layer][pos][:]
    float* vptr(int layer, int pos, const ModelConfig& cfg) {
        return v.data()
            + (size_t)layer * cfg.ctx_len * cfg.n_kv_total()
            + (size_t)pos   * cfg.n_kv_total();
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Model — moteur d'inférence transformer from scratch
//
// Usage :
//   Model m;
//   m.load("path/to/model.gguf");
//   const float* logits = m.forward(token_id, pos);
//   int next_token = ops::argmax(logits, m.cfg.n_vocab);
// ─────────────────────────────────────────────────────────────────────────
class Model {
public:
    ModelConfig cfg;

    bool load(const std::string& gguf_path);
    bool is_loaded() const { return gguf_.is_open(); }

    // Passe avant : token_id à la position pos → logits [n_vocab]
    // compute_logits=false : skip LM head (prefill) → retourne nullptr
    const float* forward(int32_t token_id, int pos, bool compute_logits = true);

    // Prefill BATCHÉ : traite les n tokens du prompt en une passe (GEMM
    // weight-stationary via gemm_q) et remplit le KV cache aux positions
    // start_pos..start_pos+n-1. Ne calcule pas de logits. Équivalent (aux
    // arrondis près) à n appels forward(..., compute_logits=false).
    void forward_prefill(const int32_t* tokens, int n, int start_pos = 0);

    // Remet le KV cache à zéro (nouvelle conversation)
    void reset() { kv_.init(cfg); }

    // Chauffe les pages de poids en RAM (transformer + LM head) puis reset.
    // À appeler une fois après load(), avant la première inférence.
    void warmup();

    // Embedding sémantique : mean-pool des hidden states sur une séquence
    // Retourne un vecteur de dimension embed_dim, utile pour le RAG sémantique
    std::vector<float> embed_tokens(const std::vector<int32_t>& tokens);

    // Snapshot/restore du KV cache (nécessaire pour le speculative decoding)
    struct KVSnapshot { KVCache kv; int pos; };
    KVSnapshot  snapshot_kv(int pos) const { return {kv_, pos}; }
    void        restore_kv(const KVSnapshot& snap) { kv_ = snap.kv; }

    // ── Persistance binaire ───────────────────────────────────────────────
    // Sauvegarde config + KV cache dans un fichier .orkv (binary).
    // pos = position courante dans la séquence (nombre de tokens traités).
    bool save_state(const std::string& path, int pos) const;

    // Charge config + KV cache depuis un fichier .orkv.
    // Remplit pos avec la position sauvegardée.
    // Retourne false si le fichier est absent ou incompatible.
    bool load_state(const std::string& path, int& pos);

    const GGUFParser& gguf() const { return gguf_; }

    // Test : vérifie que le cache de norm correspond aux poids bruts du GGUF
    // (attrape un mauvais offset/nom d'indexation). Retourne true si tout colle.
    bool test_norm_cache_matches();

private:
    GGUFParser gguf_;
    KVCache    kv_;

    // Buffers d'activation (alloués une fois, réutilisés à chaque forward)
    std::vector<float> x_;      // activation courante [embed_dim]
    std::vector<float> xb_;     // buffer [embed_dim]
    std::vector<float> xb2_;    // buffer [embed_dim]
    std::vector<float> hb_;     // buffer FFN [ffn_hidden]
    std::vector<float> hb2_;    // buffer FFN [ffn_hidden]
    std::vector<float> q_;      // requêtes  [n_heads × head_dim]
    std::vector<float> k_;      // clés      [n_kv_heads × head_dim]
    std::vector<float> v_;      // valeurs   [n_kv_heads × head_dim]
    std::vector<float> att_;    // scores attention [ctx_len]
    std::vector<float> att_scratch_; // scores par tête [n_heads × ctx_len] (remplace att_local pile)
    std::vector<float> logits_; // sorties   [n_vocab]
    std::vector<float> wbuf_;   // buffer poids dequantisés (max ffn_hidden × embed_dim)
    std::vector<float> norm_buf_; // buffer norme [embed_dim] (scratch LM head fallback)
    std::vector<float> hnorm_;  // buffer norme par tête [head_dim] (QK-norm Qwen3)

    // Poids RMSNorm mis en CACHE au chargement (constants) — évite un reload
    // (nom → lookup hash → memcpy) par token et par couche dans le forward.
    std::vector<float> attn_norm_w_;    // [n_layers × embed_dim]
    std::vector<float> ffn_norm_w_;     // [n_layers × embed_dim]
    std::vector<float> output_norm_w_;  // [embed_dim]
    std::vector<float> attn_q_norm_w_;  // [n_layers × head_dim] (Qwen3 QK-norm)
    std::vector<float> attn_k_norm_w_;  // [n_layers × head_dim]

    // Noms de tenseurs de poids par couche, construits UNE fois au chargement.
    // Évite de reconstruire "blk.N.xxx.weight" (std::to_string + concaténations,
    // ~200 allocations/token pour Qwen 7B) à chaque token et chaque couche.
    struct LayerNames {
        std::string attn_q, attn_k, attn_v, attn_output;
        std::string ffn_gate, ffn_up, ffn_down;
    };
    std::vector<LayerNames> lnames_;
    void build_layer_names();

    void alloc_buffers();
    void cache_norms();                       // pré-charge tous les poids de norm
    void load_norm(const std::string& name);  // charge un vecteur de norme F32 [embed_dim]
    // charge un vecteur de norme F32 de taille n dans dst (ex: QK-norm head_dim)
    void load_norm_n(const std::string& name, float* dst, int n);

    // Dequantise un tenseur dans wbuf_ (réutilise le buffer)
    void dq(const std::string& name);
};
