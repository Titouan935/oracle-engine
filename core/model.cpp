#include "model.hpp"
#include "bug/logger.hpp"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>

// Format binaire .orkv
// [4] magic "ORKV" | [4] version=1 | [sizeof(ModelConfig)] config
// [4] pos | [8] n_floats | n_floats×4 kv.k | n_floats×4 kv.v
static constexpr uint32_t ORKV_MAGIC   = 0x564B524F; // "ORKV" LE
static constexpr uint32_t ORKV_VERSION = 2;  // v2 : ModelConfig étendu (rms_eps, qk_norm)

// ══════════════════════════════════════════════════════════════════════════
//  Chargement du modèle depuis un fichier GGUF
// ══════════════════════════════════════════════════════════════════════════
bool Model::load(const std::string& path) {
    if (!gguf_.load(path)) return false;

    // ── Lit la configuration depuis les métadonnées GGUF ─────────────────
    // La clé varie selon l'architecture (llama, qwen3moe, etc.)
    auto arch = gguf_.meta_str("general.architecture", "llama");

    auto mk = [&](const std::string& k) { return arch + "." + k; };

    // ── Garde d'architecture ──────────────────────────────────────────────
    // Le moteur ne gère que les transformers DENSES GQA (llama, qwen2/2.5, qwen3,
    // mistral, phi). Les archis MoE (routing d'experts), SSM (selective-scan) et
    // vision-langage ne sont PAS encore supportées : sans ce garde, le forward
    // cherche des tenseurs FFN denses inexistants et produit une sortie erronée
    // silencieuse. On échoue proprement à la place. Plan : ENGINE_ARCH_SUPPORT.md.
    if (gguf_.meta_u32(mk("expert_count"), 0) > 0) {
        LOG_ERR("Model", "Architecture MoE non supportee (" + arch +
                ", expert_count>0) : routing d'experts requis. Voir ENGINE_ARCH_SUPPORT.md.");
        return false;
    }
    if (gguf_.has_meta(mk("ssm.state_size")) || gguf_.has_meta(mk("ssm.conv_kernel"))) {
        LOG_ERR("Model", "Architecture SSM non supportee (" + arch +
                ") : selective-scan (type Mamba) requis. Voir ENGINE_ARCH_SUPPORT.md.");
        return false;
    }
    if (arch.find("vl") != std::string::npos || arch.find("clip") != std::string::npos) {
        LOG_ERR("Model", "Architecture vision-langage non supportee (" + arch +
                ") : encodeur vision requis. Voir ENGINE_ARCH_SUPPORT.md.");
        return false;
    }

    cfg.n_vocab    = gguf_.meta_u32("llama.vocab_size",
                     gguf_.meta_u32(mk("vocab_size"), 32000));
    cfg.n_layers   = gguf_.meta_u32(mk("block_count"),      32);
    cfg.n_heads    = gguf_.meta_u32(mk("attention.head_count"), 32);
    cfg.n_kv_heads = gguf_.meta_u32(mk("attention.head_count_kv"),
                     cfg.n_heads);
    cfg.embed_dim  = gguf_.meta_u32(mk("embedding_length"),  4096);
    cfg.ffn_hidden = gguf_.meta_u32(mk("feed_forward_length"), 11008);
    cfg.rope_freq  = gguf_.meta_f32(mk("rope.freq_base"),  500000.f);
    // head_dim explicite dans le GGUF (Qwen3 : 128 ≠ embed_dim/n_heads).
    // Fallback embed_dim/n_heads pour Llama/Qwen2.5/Mistral/Phi.
    cfg.head_dim   = gguf_.meta_u32(mk("attention.key_length"),
                                    cfg.embed_dim / cfg.n_heads);
    cfg.rms_eps    = gguf_.meta_f32(mk("attention.layer_norm_rms_epsilon"), 1e-5f);
    // QK-norm : présent si les tenseurs de norme par tête existent (Qwen3)
    cfg.qk_norm    = gguf_.has_tensor("blk.0.attn_q_norm.weight");
    cfg.ctx_len    = 512;  // fenêtre de contexte effective (réduit le KV cache)

    // Corrige le vocab_size : priorite au tenseur output.weight (source de verite)
    // La shape peut etre [vocab, embed] ou [embed, vocab] selon le format
    if (gguf_.has_tensor("output.weight")) {
        auto& sh = gguf_.tensor_info("output.weight").shape;
        uint64_t dim0 = sh.size() > 0 ? sh[0] : 0;
        uint64_t dim1 = sh.size() > 1 ? sh[1] : 0;
        cfg.n_vocab = (uint32_t)std::max(dim0, dim1);
    } else if (gguf_.has_tensor("token_embd.weight")) {
        auto& sh = gguf_.tensor_info("token_embd.weight").shape;
        uint64_t dim0 = sh.size() > 0 ? sh[0] : 0;
        uint64_t dim1 = sh.size() > 1 ? sh[1] : 0;
        cfg.n_vocab = (uint32_t)std::max(dim0, dim1);
    } else if (cfg.n_vocab == 0) {
        cfg.n_vocab = gguf_.meta_u32("tokenizer.ggml.tokens.size", 32000);
    }

    LOG_INFO("Model", arch + " chargé — "
        + std::to_string(cfg.n_layers) + "L "
        + std::to_string(cfg.n_heads)  + "H "
        + "embed=" + std::to_string(cfg.embed_dim) + " "
        + "ffn="   + std::to_string(cfg.ffn_hidden) + " "
        + "hd="    + std::to_string(cfg.head_dim) + " "
        + "vocab=" + std::to_string(cfg.n_vocab)
        + (cfg.qk_norm ? " [qk_norm]" : ""));

    kv_.init(cfg);
    alloc_buffers();
    cache_norms();        // pré-charge les poids RMSNorm (constants) une fois
    build_layer_names();  // construit les noms de tenseurs par couche une fois
    return true;
}

// ── Construit les noms de tenseurs de poids par couche (une seule fois) ─────
void Model::build_layer_names() {
    lnames_.resize(cfg.n_layers);
    for (int l = 0; l < (int)cfg.n_layers; l++) {
        const std::string b = "blk." + std::to_string(l) + ".";
        LayerNames& n = lnames_[l];
        n.attn_q      = b + "attn_q.weight";
        n.attn_k      = b + "attn_k.weight";
        n.attn_v      = b + "attn_v.weight";
        n.attn_output = b + "attn_output.weight";
        n.ffn_gate    = b + "ffn_gate.weight";
        n.ffn_up      = b + "ffn_up.weight";
        n.ffn_down    = b + "ffn_down.weight";
    }
}

// ── Pré-charge tous les poids de RMSNorm (constants) en cache ──────────────
// Utilise EXACTEMENT les mêmes noms de tenseurs que le forward → résultat
// numériquement identique, mais sans reload par token.
void Model::cache_norms() {
    const int D = (int)cfg.embed_dim, L = (int)cfg.n_layers, Hd = (int)cfg.head_dim;
    attn_norm_w_.resize((size_t)L * D);
    ffn_norm_w_.resize((size_t)L * D);
    output_norm_w_.resize(D);
    for (int l = 0; l < L; l++) {
        auto ln = [&](const std::string& k) { return "blk." + std::to_string(l) + "." + k; };
        load_norm_n(ln("attn_norm.weight"), &attn_norm_w_[(size_t)l * D], D);
        load_norm_n(ln("ffn_norm.weight"),  &ffn_norm_w_[(size_t)l * D], D);
    }
    load_norm_n("output_norm.weight", output_norm_w_.data(), D);

    if (cfg.qk_norm) {
        attn_q_norm_w_.resize((size_t)L * Hd);
        attn_k_norm_w_.resize((size_t)L * Hd);
        for (int l = 0; l < L; l++) {
            auto ln = [&](const std::string& k) { return "blk." + std::to_string(l) + "." + k; };
            load_norm_n(ln("attn_q_norm.weight"), &attn_q_norm_w_[(size_t)l * Hd], Hd);
            load_norm_n(ln("attn_k_norm.weight"), &attn_k_norm_w_[(size_t)l * Hd], Hd);
        }
    }
}

bool Model::save_state(const std::string& path, int pos) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { LOG_ERR("Model", "save_state: impossible d'ouvrir " + path); return false; }

    uint32_t ipos = (uint32_t)pos;
    uint64_t n    = (uint64_t)kv_.k.size();
    bool ok = true;
    ok = ok && std::fwrite(&ORKV_MAGIC,   sizeof(uint32_t),    1, f) == 1;
    ok = ok && std::fwrite(&ORKV_VERSION, sizeof(uint32_t),    1, f) == 1;
    ok = ok && std::fwrite(&cfg,          sizeof(ModelConfig), 1, f) == 1;
    ok = ok && std::fwrite(&ipos,         sizeof(uint32_t),    1, f) == 1;
    ok = ok && std::fwrite(&n,            sizeof(uint64_t),    1, f) == 1;
    ok = ok && std::fwrite(kv_.k.data(),  sizeof(float),  n, f) == n;
    ok = ok && std::fwrite(kv_.v.data(),  sizeof(float),  n, f) == n;

    std::fclose(f);
    if (!ok) {
        LOG_ERR("Model", "save_state: écriture incomplète (disque plein ?) → " + path);
        return false;
    }
    LOG_INFO("Model", "État sauvegardé → " + path
             + " (pos=" + std::to_string(pos)
             + " kv=" + std::to_string(n * 2 * 4 / 1024 / 1024) + " MB)");
    return true;
}

bool Model::load_state(const std::string& path, int& pos) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic = 0, ver = 0;
    bool ok = std::fread(&magic, sizeof(uint32_t), 1, f) == 1
           && std::fread(&ver,   sizeof(uint32_t), 1, f) == 1;
    if (!ok || magic != ORKV_MAGIC || ver != ORKV_VERSION) {
        LOG_ERR("Model", "load_state: fichier incompatible " + path);
        std::fclose(f); return false;
    }

    ModelConfig loaded_cfg;
    if (std::fread(&loaded_cfg, sizeof(ModelConfig), 1, f) != 1) {
        LOG_ERR("Model", "load_state: en-tête tronqué " + path);
        std::fclose(f); return false;
    }
    if (loaded_cfg.n_layers != cfg.n_layers || loaded_cfg.embed_dim != cfg.embed_dim
        || loaded_cfg.ctx_len != cfg.ctx_len || loaded_cfg.n_kv_heads != cfg.n_kv_heads
        || loaded_cfg.head_dim != cfg.head_dim) {
        LOG_ERR("Model", "load_state: config incompatible avec le modèle chargé "
                "(layers/embed/ctx/kv_heads/head_dim)");
        std::fclose(f); return false;
    }

    uint32_t ipos = 0;
    uint64_t n = 0;
    if (std::fread(&ipos, sizeof(uint32_t), 1, f) != 1
        || std::fread(&n, sizeof(uint64_t), 1, f) != 1) {
        LOG_ERR("Model", "load_state: en-tête tronqué " + path);
        std::fclose(f); return false;
    }
    // La taille KV doit correspondre au layout attendu pour la config courante.
    const uint64_t expected = (uint64_t)cfg.n_layers * cfg.ctx_len * cfg.n_kv_total();
    if (n != expected) {
        LOG_ERR("Model", "load_state: taille KV incohérente (" + std::to_string(n)
                + " vs " + std::to_string(expected) + ")");
        std::fclose(f); return false;
    }
    pos = (int)ipos;

    kv_.k.resize(n);
    kv_.v.resize(n);
    if (std::fread(kv_.k.data(), sizeof(float), n, f) != n
        || std::fread(kv_.v.data(), sizeof(float), n, f) != n) {
        LOG_ERR("Model", "load_state: données KV tronquées " + path);
        std::fclose(f); return false;
    }

    std::fclose(f);
    LOG_INFO("Model", "État chargé ← " + path
             + " (pos=" + std::to_string(pos)
             + " kv=" + std::to_string(n * 2 * 4 / 1024 / 1024) + " MB)");
    return true;
}

void Model::warmup() {
    // Deux passes forward incluant le LM head pour établir le working set complet
    // (transformer + output.weight) dans un état stable avant la vraie inférence.
    for (int i = 0; i < 2; i++) {
        forward(0, 0, /*compute_logits=*/true);
        kv_.init(cfg);
    }
}

// ── Embedding sémantique (mean pool des hidden states) ───────────────────────
std::vector<float> Model::embed_tokens(const std::vector<int32_t>& tokens) {
    if (tokens.empty() || !is_loaded()) return {};
    std::vector<float> acc(cfg.embed_dim, 0.f);
    kv_.init(cfg);
    int n = (int)tokens.size();
    for (int i = 0; i < n; i++) {
        // forward sans LM head — x_ contient les hidden states après la dernière couche
        forward(tokens[i], i, /*compute_logits=*/false);
        for (int d = 0; d < (int)cfg.embed_dim; d++)
            acc[d] += x_[d];
    }
    // Mean pooling + L2 normalisation
    float inv_n = 1.f / n;
    float norm2 = 0.f;
    for (int d = 0; d < (int)cfg.embed_dim; d++) {
        acc[d] *= inv_n;
        norm2 += acc[d] * acc[d];
    }
    float inv_norm = 1.f / (std::sqrt(norm2) + 1e-9f);
    for (float& v : acc) v *= inv_norm;
    kv_.init(cfg);  // remet le KV à zéro après l'embedding
    return acc;
}

// ── Pré-allocation des buffers d'activation ───────────────────────────────
void Model::alloc_buffers() {
    x_.resize(cfg.embed_dim);
    // xb_ sert aussi d'accumulateur d'attention [n_heads × head_dim] :
    // pour Qwen3 n_q_total() (2048) > embed_dim (1024).
    xb_.resize(std::max(cfg.embed_dim, cfg.n_q_total()));
    xb2_.resize(cfg.embed_dim);
    hb_.resize(cfg.ffn_hidden);
    hb2_.resize(cfg.ffn_hidden);
    q_.resize(cfg.n_q_total());
    k_.resize(cfg.n_kv_total());
    v_.resize(cfg.n_kv_total());
    att_.resize(cfg.ctx_len);
    // Scratch d'attention par tête : [n_heads × ctx_len]. Remplace le buffer de
    // pile att_local[512] → sûr même si ctx_len dépasse 512.
    att_scratch_.resize((size_t)cfg.n_heads * cfg.ctx_len);
    logits_.resize(cfg.n_vocab);
    // Buffer poids : taille du plus grand tenseur (ffn_gate/up : ffn_hidden × embed_dim)
    wbuf_.resize((size_t)cfg.ffn_hidden * cfg.embed_dim);
    norm_buf_.resize(cfg.embed_dim);
    hnorm_.resize(cfg.head_dim);
}

// ── Test : le cache de norm == poids bruts du GGUF ? ──────────────────────
bool Model::test_norm_cache_matches() {
    const int D = (int)cfg.embed_dim, L = (int)cfg.n_layers, Hd = (int)cfg.head_dim;
    std::vector<float> tmp(std::max(D, Hd));
    auto same = [&](const std::string& name, const float* cached, int n) {
        load_norm_n(name, tmp.data(), n);
        for (int i = 0; i < n; i++) if (tmp[i] != cached[i]) return false;
        return true;
    };
    for (int l = 0; l < L; l++) {
        auto ln = [&](const std::string& k) { return "blk." + std::to_string(l) + "." + k; };
        if (!same(ln("attn_norm.weight"), &attn_norm_w_[(size_t)l * D], D)) return false;
        if (!same(ln("ffn_norm.weight"),  &ffn_norm_w_[(size_t)l * D], D)) return false;
        if (cfg.qk_norm) {
            if (!same(ln("attn_q_norm.weight"), &attn_q_norm_w_[(size_t)l * Hd], Hd)) return false;
            if (!same(ln("attn_k_norm.weight"), &attn_k_norm_w_[(size_t)l * Hd], Hd)) return false;
        }
    }
    return same("output_norm.weight", output_norm_w_.data(), D);
}

// ── Charge un vecteur de norme F32 de taille n dans dst ───────────────────
void Model::load_norm_n(const std::string& name, float* dst, int n) {
    const uint8_t* raw = gguf_.tensor_raw(name);
    if (raw) std::memcpy(dst, raw, (size_t)n * 4);
}

// ── Charge un vecteur de norme F32 [embed_dim] dans norm_buf_ ─────────────
void Model::load_norm(const std::string& name) {
    load_norm_n(name, norm_buf_.data(), (int)cfg.embed_dim);
}

// ── Dequantise un tenseur dans wbuf_ ─────────────────────────────────────
void Model::dq(const std::string& name) {
    gguf_.dequantize_into(name, wbuf_);
}

// ══════════════════════════════════════════════════════════════════════════
//  Passe avant — un token à la fois (génération autoregressive)
//
//  token_id : ID du token courant (0-indexed dans le vocabulaire)
//  pos      : position dans la séquence (0-indexed)
//
//  Retourne un pointeur vers logits_[0..n_vocab-1]
// ══════════════════════════════════════════════════════════════════════════
const float* Model::forward(int32_t token_id, int pos, bool compute_logits) {
    const int D  = (int)cfg.embed_dim;
    const int H  = (int)cfg.n_heads;
    const int KVH= (int)cfg.n_kv_heads;
    const int Hd = (int)cfg.head_dim;
    const int FF = (int)cfg.ffn_hidden;
    const int NQ = (int)cfg.n_q_total();
    const int NKV= (int)cfg.n_kv_total();

    // ── 1. Embedding lookup ───────────────────────────────────────────────
    // Garde : un token_id hors plage ferait une lecture hors limites dans les
    // poids d'embedding. On le ramène dans [0, n_vocab).
    if (token_id < 0 || token_id >= (int)cfg.n_vocab) {
        LOG_WARN("Model", "token_id hors plage (" + std::to_string(token_id)
                 + "), ramené à 0");
        token_id = 0;
    }
    // Dequantise seulement la ligne token_id de token_embd.weight
    gguf_.dequantize_row("token_embd.weight", token_id, x_.data());

    // ── 2. Boucle sur les couches transformer ─────────────────────────────
    for (int l = 0; l < (int)cfg.n_layers; l++) {
        const LayerNames& ln = lnames_[l];

        // ─── Self-Attention ───────────────────────────────────────────────

        // RMSNorm d'entrée (poids en cache — pas de reload par token)
        ops::rmsnorm(xb_.data(), x_.data(), &attn_norm_w_[(size_t)l * D], D);

        // Projections Q, K, V   — GEMV direct sur données quantisées (sans wbuf_)
        if (!gguf_.gemv_q(ln.attn_q,  q_.data(), xb_.data(), NQ,  D)) {
            dq(ln.attn_q);
            ops::matmul(q_.data(), xb_.data(), wbuf_.data(), 1, NQ, D);
        }
        if (!gguf_.gemv_q(ln.attn_k,  k_.data(), xb_.data(), NKV, D)) {
            dq(ln.attn_k);
            ops::matmul(k_.data(), xb_.data(), wbuf_.data(), 1, NKV, D);
        }
        if (!gguf_.gemv_q(ln.attn_v,  v_.data(), xb_.data(), NKV, D)) {
            dq(ln.attn_v);
            ops::matmul(v_.data(), xb_.data(), wbuf_.data(), 1, NKV, D);
        }

        // QK-norm (Qwen3) : RMSNorm par tête sur head_dim, avant RoPE.
        // Poids partagés entre toutes les têtes (taille head_dim).
        if (cfg.qk_norm) {
            const float* qn = &attn_q_norm_w_[(size_t)l * Hd];   // poids en cache
            for (int h = 0; h < H; h++)
                ops::rmsnorm(q_.data() + h * Hd, q_.data() + h * Hd, qn, Hd, cfg.rms_eps);
            const float* kn = &attn_k_norm_w_[(size_t)l * Hd];
            for (int h = 0; h < KVH; h++)
                ops::rmsnorm(k_.data() + h * Hd, k_.data() + h * Hd, kn, Hd, cfg.rms_eps);
        }

        // RoPE — encodage de position rotatif sur Q et K
        ops::rope(q_.data(), k_.data(), H, KVH, Hd, pos, cfg.rope_freq);

        // Stocke K et V dans le cache
        float* kc = kv_.kptr(l, pos, cfg);
        float* vc = kv_.vptr(l, pos, cfg);
        ops::copy(kc, k_.data(), NKV);
        ops::copy(vc, v_.data(), NKV);

        // Multi-head attention (GQA) — chaque tête est indépendante → OpenMP
        std::fill(xb_.begin(), xb_.end(), 0.f);
        const float scale = 1.f / std::sqrt((float)Hd);
        const int n_gqa   = (int)cfg.n_gqa();

        #pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            const int kv_h = h / n_gqa;
            const float* qh = q_.data() + h * Hd;
            float*       oh = xb_.data() + h * Hd;  // plage non-chevauchante

            // Scratch d'attention par tête (plage non-chevauchante → pas de
            // contention, et sûr quel que soit ctx_len).
            float* att_local = att_scratch_.data() + (size_t)h * cfg.ctx_len;
            for (int t = 0; t <= pos; t++) {
                const float* kh = kv_.kptr(l, t, cfg) + kv_h * Hd;
                att_local[t] = ops::dot(qh, kh, Hd) * scale;
            }
            ops::softmax(att_local, pos + 1);

            for (int t = 0; t <= pos; t++) {
                const float* vh = kv_.vptr(l, t, cfg) + kv_h * Hd;
                const float  w  = att_local[t];
                for (int i = 0; i < Hd; i++) oh[i] += w * vh[i];
            }
        }

        // Projection de sortie de l'attention  (xb_ → xb2_)
        if (!gguf_.gemv_q(ln.attn_output, xb2_.data(), xb_.data(), D, NQ)) {
            dq(ln.attn_output);
            ops::matmul(xb2_.data(), xb_.data(), wbuf_.data(), 1, D, NQ);
        }

        // Connexion résiduelle
        ops::add(x_.data(), xb2_.data(), D);

        // ─── Feed-Forward (SwiGLU) ────────────────────────────────────────

        // RMSNorm d'entrée FFN (poids en cache)
        ops::rmsnorm(xb_.data(), x_.data(), &ffn_norm_w_[(size_t)l * D], D);

        // Gate : hb_ = xb_ @ W_gate^T
        if (!gguf_.gemv_q(ln.ffn_gate, hb_.data(),  xb_.data(), FF, D)) {
            dq(ln.ffn_gate);
            ops::matmul(hb_.data(), xb_.data(), wbuf_.data(), 1, FF, D);
        }
        // Up   : hb2_ = xb_ @ W_up^T
        if (!gguf_.gemv_q(ln.ffn_up,   hb2_.data(), xb_.data(), FF, D)) {
            dq(ln.ffn_up);
            ops::matmul(hb2_.data(), xb_.data(), wbuf_.data(), 1, FF, D);
        }

        // SwiGLU : hb_ = SiLU(hb_) ⊙ hb2_
        ops::silu(hb_.data(), FF);
        ops::mul(hb_.data(), hb_.data(), hb2_.data(), FF);

        // Down : xb_ = hb_ @ W_down^T
        if (!gguf_.gemv_q(ln.ffn_down, xb_.data(), hb_.data(), D, FF)) {
            dq(ln.ffn_down);
            ops::matmul(xb_.data(), hb_.data(), wbuf_.data(), 1, D, FF);
        }

        // Connexion résiduelle
        ops::add(x_.data(), xb_.data(), D);
    }

    // ── 3. Normalisation finale ───────────────────────────────────────────
    if (!compute_logits) return nullptr;

    ops::rmsnorm(x_.data(), x_.data(), output_norm_w_.data(), D);   // poids en cache

    // ── 4. Tête de langage (LM head) → logits (parallélisée via OpenMP) ─────
    const std::string lm_head = gguf_.has_tensor("output.weight")
                               ? "output.weight"
                               : "token_embd.weight";
    if (!gguf_.gemv_q(lm_head, logits_.data(), x_.data(), cfg.n_vocab, D)) {
        // Fallback F32/BF16 — boucle scalaire
        norm_buf_.resize(D);
        for (int i = 0; i < (int)cfg.n_vocab; i++) {
            gguf_.dequantize_row(lm_head, i, norm_buf_.data());
            logits_[i] = ops::dot(x_.data(), norm_buf_.data(), D);
        }
    }

    return logits_.data();
}

// ══════════════════════════════════════════════════════════════════════════
//  forward_prefill — traite n tokens du prompt EN BATCH.
//
//  Différence clé avec forward() (single-token) : les projections Q/K/V/out
//  et FFN passent par gemm_q (weight-stationary). Chaque ligne de poids est
//  déquantisée UNE fois puis appliquée aux n tokens → le coût de dequant est
//  amorti sur tout le prompt au lieu d'être payé n fois. C'est LE gain du
//  prefill batché : le décodage est memory-bound, donc relire les poids une
//  seule fois pour n tokens divise le trafic mémoire de dequant par ~n.
//
//  Résultat identique (aux arrondis flottants près) à n appels forward().
//  Ne touche pas aux buffers single-token (x_, q_, …) : tout est local.
// ══════════════════════════════════════════════════════════════════════════
void Model::forward_prefill(const int32_t* tokens, int n, int start_pos) {
    if (n <= 0) return;
    const int D   = (int)cfg.embed_dim;
    const int H   = (int)cfg.n_heads;
    const int KVH = (int)cfg.n_kv_heads;
    const int Hd  = (int)cfg.head_dim;
    const int FF  = (int)cfg.ffn_hidden;
    const int NQ  = (int)cfg.n_q_total();
    const int NKV = (int)cfg.n_kv_total();
    const int n_gqa = (int)cfg.n_gqa();
    const float scale = 1.f / std::sqrt((float)Hd);

    // Buffers batchés locaux (prefill appelé une fois par génération).
    std::vector<float> X ((size_t)n * D);    // flux résiduel [n × D]
    std::vector<float> Xn((size_t)n * D);    // normalisé
    std::vector<float> Q ((size_t)n * NQ);
    std::vector<float> K ((size_t)n * NKV);
    std::vector<float> V ((size_t)n * NKV);
    std::vector<float> AO((size_t)n * NQ);   // sortie attention (pré-projection)
    std::vector<float> XO((size_t)n * D);    // sortie projection (attn + ffn down)
    std::vector<float> G ((size_t)n * FF);   // gate FFN
    std::vector<float> U ((size_t)n * FF);   // up FFN
    // Scratch d'attention par tête [n_heads × ctx_len] (sûr quel que soit ctx_len).
    std::vector<float> att_scratch((size_t)H * cfg.ctx_len);

    // Helper : projection GEMM avec repli per-token gemv_q si gemm_q échoue.
    auto proj = [&](const std::string& name, float* out, const float* in,
                    int n_out, int n_in) {
        if (gguf_.gemm_q(name, out, in, n, n_out, n_in)) return;
        for (int t = 0; t < n; t++)
            if (!gguf_.gemv_q(name, out + (size_t)t * n_out,
                              in + (size_t)t * n_in, n_out, n_in)) {
                dq(name);
                ops::matmul(out + (size_t)t * n_out, in + (size_t)t * n_in,
                            wbuf_.data(), 1, n_out, n_in);
            }
    };

    // 1. Embeddings (une ligne par token). Garde token_id hors plage.
    for (int t = 0; t < n; t++) {
        int32_t tok = tokens[t];
        if (tok < 0 || tok >= (int32_t)cfg.n_vocab) {
            LOG_WARN("Model", "forward_prefill: token_id hors plage ("
                     + std::to_string(tok) + "), ramené à 0");
            tok = 0;
        }
        gguf_.dequantize_row("token_embd.weight", tok, X.data() + (size_t)t * D);
    }

    // 2. Couches transformer.
    for (int l = 0; l < (int)cfg.n_layers; l++) {
        const LayerNames& ln = lnames_[l];

        // RMSNorm attn (poids en cache), par token.
        for (int t = 0; t < n; t++)
            ops::rmsnorm(Xn.data() + (size_t)t * D, X.data() + (size_t)t * D,
                         &attn_norm_w_[(size_t)l * D], D);

        // Projections Q, K, V — GEMM batché.
        proj(ln.attn_q, Q.data(), Xn.data(), NQ,  D);
        proj(ln.attn_k, K.data(), Xn.data(), NKV, D);
        proj(ln.attn_v, V.data(), Xn.data(), NKV, D);

        // QK-norm (Qwen3), par token puis par tête.
        if (cfg.qk_norm) {
            const float* qn = &attn_q_norm_w_[(size_t)l * Hd];
            const float* kn = &attn_k_norm_w_[(size_t)l * Hd];
            for (int t = 0; t < n; t++) {
                float* qt = Q.data() + (size_t)t * NQ;
                float* kt = K.data() + (size_t)t * NKV;
                for (int h = 0; h < H;   h++) ops::rmsnorm(qt + h * Hd, qt + h * Hd, qn, Hd, cfg.rms_eps);
                for (int h = 0; h < KVH; h++) ops::rmsnorm(kt + h * Hd, kt + h * Hd, kn, Hd, cfg.rms_eps);
            }
        }

        // RoPE + stockage KV, chaque token à sa position absolue.
        for (int t = 0; t < n; t++) {
            const int p = start_pos + t;
            ops::rope(Q.data() + (size_t)t * NQ, K.data() + (size_t)t * NKV,
                      H, KVH, Hd, p, cfg.rope_freq);
            ops::copy(kv_.kptr(l, p, cfg), K.data() + (size_t)t * NKV, NKV);
            ops::copy(kv_.vptr(l, p, cfg), V.data() + (size_t)t * NKV, NKV);
        }

        // Attention causale : token t attend 0..start_pos+t. Têtes en OpenMP.
        for (int t = 0; t < n; t++) {
            const int p = start_pos + t;
            float* ao_t = AO.data() + (size_t)t * NQ;
            std::fill(ao_t, ao_t + NQ, 0.f);
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < H; h++) {
                const int kv_h = h / n_gqa;
                const float* qh = Q.data() + (size_t)t * NQ + h * Hd;
                float*       oh = ao_t + h * Hd;
                float* att_local = att_scratch.data() + (size_t)h * cfg.ctx_len;
                for (int s = 0; s <= p; s++) {
                    const float* kh = kv_.kptr(l, s, cfg) + kv_h * Hd;
                    att_local[s] = ops::dot(qh, kh, Hd) * scale;
                }
                ops::softmax(att_local, p + 1);
                for (int s = 0; s <= p; s++) {
                    const float* vh = kv_.vptr(l, s, cfg) + kv_h * Hd;
                    const float  w  = att_local[s];
                    for (int i = 0; i < Hd; i++) oh[i] += w * vh[i];
                }
            }
        }

        // Projection sortie attn + résiduel.
        proj(ln.attn_output, XO.data(), AO.data(), D, NQ);
        for (size_t i = 0; i < (size_t)n * D; i++) X[i] += XO[i];

        // FFN SwiGLU.
        for (int t = 0; t < n; t++)
            ops::rmsnorm(Xn.data() + (size_t)t * D, X.data() + (size_t)t * D,
                         &ffn_norm_w_[(size_t)l * D], D);
        proj(ln.ffn_gate, G.data(), Xn.data(), FF, D);
        proj(ln.ffn_up,   U.data(), Xn.data(), FF, D);
        for (size_t i = 0; i < (size_t)n * FF; i++) {
            const float g = G[i];
            G[i] = (g / (1.f + std::exp(-g))) * U[i];   // SiLU(g) ⊙ up
        }
        proj(ln.ffn_down, XO.data(), G.data(), D, FF);
        for (size_t i = 0; i < (size_t)n * D; i++) X[i] += XO[i];
    }
    // Prefill : pas de logits. Le KV cache est rempli pour start_pos..start_pos+n-1.
}
