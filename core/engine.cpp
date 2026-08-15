#include "engine.hpp"
#include "bug/logger.hpp"
#include "core/machine_profile.hpp"
#include <chrono>
#include <cstdint>
#include <cstring>

bool Engine::load(const std::string& path, const GenParams&) {
    // Le core s'adapte à la machine : détecte CPU/SIMD/RAM et règle les threads
    // d'inférence pour exploiter le LLM au maximum sur cette machine précise.
    MachineProfile mp = detect_machine();
    apply_machine_profile(mp);

    if (!model_.load(path))    { LOG_ERR("Engine", "load modèle échoué"); return false; }
    if (!tok_.load(model_.gguf())) { LOG_ERR("Engine", "load tokenizer échoué"); return false; }
    model_path_ = path;
    if (model_.gguf().is_mmap_only()) {
        LOG_INFO("Engine", "Warmup skip (mmap-only, RAM insuffisante)");
    } else {
        LOG_INFO("Engine", "Warmup RAM...");
        model_.warmup();
    }
    logits_buf_.resize(model_.cfg.n_vocab);
    LOG_INFO("Engine", "Prêt — " + std::to_string(model_.cfg.n_layers) + "L "
             + std::to_string(model_.cfg.n_vocab) + " vocab");
    return true;
}

bool Engine::load_draft(const std::string& draft_path) {
    if (!draft_model_.load(draft_path)) {
        LOG_ERR("Engine", "load draft échoué : " + draft_path);
        return false;
    }
    if (!draft_tok_.load(draft_model_.gguf())) {
        LOG_ERR("Engine", "load draft tokenizer échoué");
        return false;
    }
    if (!draft_model_.gguf().is_mmap_only()) {
        draft_model_.warmup();
    }
    draft_loaded_ = true;
    LOG_INFO("Engine", "Draft chargé — " + std::to_string(draft_model_.cfg.n_layers) + "L");
    return true;
}

std::vector<float> Engine::embed(const std::string& text) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!model_.is_loaded()) return {};
    auto ids = tok_.encode(text, /*add_bos=*/true);
    if (ids.empty()) return {};
    if (ids.size() > 512) ids.resize(512); // limite embedding à 512 tokens
    return model_.embed_tokens(ids);
}

std::string Engine::format_prompt(const std::vector<Message>& messages) const {
    const std::string arch = model_.gguf().meta_str("general.architecture", "llama");
    const bool is_qwen = (arch.size() >= 4 && arch.substr(0, 4) == "qwen");

    bool has_system = false;
    for (const auto& m : messages)
        if (m.role == "system") { has_system = true; break; }

    std::string p;
    if (is_qwen) {
        // ChatML — Qwen2 / Qwen2.5
        if (!has_system)
            p = "<|im_start|>system\nTu es un assistant utile.<|im_end|>\n";
        for (const auto& m : messages)
            p += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        p += "<|im_start|>assistant\n";
    } else {
        // Llama 3 instruct (défaut pour llama, mistral, phi, etc.)
        p = "<|begin_of_text|>";
        if (!has_system)
            p += "<|start_header_id|>system<|end_header_id|>\n\nTu es un assistant utile.<|eot_id|>";
        for (const auto& m : messages)
            p += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
        p += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return p;
}

std::string Engine::generate(const std::vector<Message>& messages,
                              const GenParams& params,
                              std::function<void(const std::string&)> on_token) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!model_.is_loaded()) return "";

    // Encode le prompt complet (BOS ajouté automatiquement)
    const std::string prompt = format_prompt(messages);
    auto ids = tok_.encode(prompt, /*add_bos=*/true);

    const int max_ctx = (int)model_.cfg.ctx_len;
    if ((int)ids.size() >= max_ctx - 4) {
        LOG_WARN("Engine", "Prompt trop long (" + std::to_string(ids.size())
                 + " tokens / ctx=" + std::to_string(max_ctx) + ")");
        return "";
    }

    model_.reset();

    // Paramètres de sampling
    SamplerParams sp;
    sp.temperature = params.temperature;
    sp.top_p       = params.top_p;
    sp.top_k       = 40;
    sp.repeat_pen  = params.repeat_penalty;
    sp.repeat_last = params.repeat_last_n;
    const uint64_t seed = (uint64_t)std::chrono::steady_clock::now()
                              .time_since_epoch().count();
    Sampler sampler(sp, seed);

    // Prefill BATCHÉ : tous les tokens du prompt sauf le dernier, en une passe.
    // forward_prefill remplit le KV cache (positions 0..n_prefill-1) via un GEMM
    // weight-stationary : chaque poids est déquantisé UNE fois et appliqué à tous
    // les tokens du prompt, au lieu d'une relecture par token (décodage
    // memory-bound → gain ≈ ×n_prefill sur le trafic mémoire de dequant).
    int pos = 0;
    const int n_prefill = (int)ids.size() - 1;
    if (n_prefill > 0) {
        model_.forward_prefill(ids.data(), n_prefill, 0);
        for (int i = 0; i < n_prefill; i++) sampler.record(ids[i]);
        pos = n_prefill;
    }

    // Génération autoregressive
    int32_t next = ids.back();
    const std::string arch2 = model_.gguf().meta_str("general.architecture", "llama");
    const bool is_qwen2 = (arch2.size() >= 4 && arch2.substr(0, 4) == "qwen");
    int32_t eos = tok_.special_id(is_qwen2 ? "<|im_end|>" : "<|eot_id|>");
    if (eos < 0) eos = tok_.special_id(is_qwen2 ? "<|eot_id|>" : "<|im_end|>");
    std::string resp;

    if (!draft_loaded_) {
        // ── Génération standard ───────────────────────────────────────────
        for (int step = 0; step < params.n_predict && pos < max_ctx - 1; step++) {
            const float* logits = model_.forward(next, pos);
            pos++;
            if (!logits) break;

            std::memcpy(logits_buf_.data(), logits, model_.cfg.n_vocab * sizeof(float));
            next = sampler.sample(logits_buf_.data(), (int)model_.cfg.n_vocab);

            if (eos >= 0 && next == eos) break;
            std::string piece = tok_.decode_one(next);
            resp += piece;
            if (on_token) on_token(piece);
        }
    } else {
        // ── Speculative decoding ─────────────────────────────────────────────
        // Algorithme :
        //  a) draft propose SPEC_N tokens en greedy
        //  b) main vérifie chaque token séquentiellement
        //  c) accepte le préfixe commun, sort le premier désaccord depuis main
        //  d) sync incrémentale du draft (O(1) au lieu de O(n) replay)
        //
        // Le KV cache du draft reste valide après rejet car les positions
        // acceptées contiennent les mêmes tokens que le main. Les positions
        // au-delà seront écrasées par la prochaine vague.

        // Prefill initial du draft model — batché (même gain weight-stationary)
        draft_model_.reset();
        if ((int)ids.size() - 1 > 0)
            draft_model_.forward_prefill(ids.data(), (int)ids.size() - 1, 0);

        for (int step = 0; step < params.n_predict && pos < max_ctx - 1; ) {
            // ── Étape a : draft génère SPEC_N tokens greedily ────────────────
            std::vector<int32_t> draft_tokens;
            {
                int dp = pos;
                int32_t dt = next;
                for (int k = 0; k < SPEC_N && dp < (int)draft_model_.cfg.ctx_len - 1; k++) {
                    const float* dl = draft_model_.forward(dt, dp++);
                    if (!dl) break;
                    int32_t best = 0;
                    const int dv = (int)draft_model_.cfg.n_vocab;
                    for (int v = 1; v < dv; v++)
                        if (dl[v] > dl[best]) best = v;
                    draft_tokens.push_back(best);
                    dt = best;
                }
            }
            if (draft_tokens.empty()) break;

            // ── Étape b : main vérifie chaque draft token ─────────────────────
            bool done = false;
            int accepted = 0;
            int32_t verify = next;
            for (int k = 0; k < (int)draft_tokens.size(); k++) {
                if (pos >= max_ctx - 1) { done = true; break; }
                const float* logits = model_.forward(verify, pos++);
                if (!logits) { done = true; break; }
                std::memcpy(logits_buf_.data(), logits, model_.cfg.n_vocab * sizeof(float));
                int32_t main_tok = sampler.sample(logits_buf_.data(), (int)model_.cfg.n_vocab);
                sampler.record(main_tok);

                if (main_tok != draft_tokens[k]) {
                    next = main_tok;
                    if (eos >= 0 && next == eos) { done = true; break; }
                    std::string piece = tok_.decode_one(next);
                    resp += piece;
                    if (on_token) on_token(piece);
                    step++;
                    break;
                }
                next = draft_tokens[k];
                accepted++;
                if (eos >= 0 && next == eos) { done = true; break; }
                std::string piece = tok_.decode_one(next);
                resp += piece;
                if (on_token) on_token(piece);
                step++;
                if (step >= params.n_predict) { done = true; break; }
                verify = next;
            }
            if (done) break;

            // ── Étape c : bonus token si tous acceptés ────────────────────────
            if (accepted == (int)draft_tokens.size() && pos < max_ctx - 1) {
                const float* logits = model_.forward(next, pos++);
                if (!logits) break;
                std::memcpy(logits_buf_.data(), logits, model_.cfg.n_vocab * sizeof(float));
                next = sampler.sample(logits_buf_.data(), (int)model_.cfg.n_vocab);
                sampler.record(next);
                if (eos >= 0 && next == eos) break;
                std::string piece = tok_.decode_one(next);
                resp += piece;
                if (on_token) on_token(piece);
                step++;

                // Sync draft : forward le dernier token accepté pour combler
                // le trou dans le KV cache du draft (step a n'allait que
                // jusqu'à pos-2, cette position manquait)
                draft_model_.forward(draft_tokens.back(), pos - 1, false);
            }
            // Rejet partiel : pas de resync nécessaire. Le prochain forward du
            // draft à pos écrasera l'entrée KV invalide, et les positions
            // antérieures sont correctes (mêmes tokens que main).
        }
    }

    return resp;
}
