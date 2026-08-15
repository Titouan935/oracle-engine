#pragma once
#include "model.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"
#include <string>
#include <vector>
#include <functional>
#include <mutex>

// Chemin par défaut du modèle principal (Qwen2.5 7B Instruct Q4_K_M)
static constexpr const char* ORACLE_DEFAULT_MODEL =
    "models/qwen2.5-7b-instruct-q4km.gguf";

// Chemin par défaut du modèle draft (Llama 3.2 3B — speculative decoding)
static constexpr const char* ORACLE_DEFAULT_DRAFT =
    "models/llama-3.2-3b-q4k.gguf";

// Interface identique à v1/core/engine.hpp — drop-in replacement sans llama.cpp
struct Message {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
};
using ChatMessage = Message;

struct GenParams {
    int   n_predict      = 256;
    float temperature    = 0.7f;
    float top_p          = 0.9f;
    float repeat_penalty = 1.15f;
    int   repeat_last_n  = 64;
    int   n_gpu_layers   = 0;   // ignoré (moteur CPU)
};

class Engine {
public:
    Engine()  = default;
    ~Engine() = default;

    // Charge le modèle GGUF + tokenizer, effectue le warmup RAM
    bool load(const std::string& model_path, const GenParams& params = {});

    // Génère une réponse complète pour une conversation (thread-safe)
    // on_token : appelé pour chaque token produit (streaming)
    std::string generate(const std::vector<Message>& messages,
                         const GenParams& params = {},
                         std::function<void(const std::string&)> on_token = nullptr);

    // Calcule un vecteur d'embedding sémantique pour un texte (pour le RAG)
    std::vector<float> embed(const std::string& text);

    // Charge un modèle draft pour le speculative decoding (facultatif)
    bool load_draft(const std::string& draft_path);
    bool has_draft() const { return draft_loaded_; }

    bool        is_loaded()  const { return model_.is_loaded(); }
    std::string model_path() const { return model_path_; }
    Tokenizer&  tokenizer()        { return tok_; }

private:
    std::string model_path_;
    Model       model_;
    Tokenizer   tok_;
    std::mutex  mtx_;

    // Buffer logits pré-alloué (évite ~600KB d'allocation par token pour Qwen 7B)
    std::vector<float> logits_buf_;

    // Draft model pour le speculative decoding
    Model     draft_model_;
    Tokenizer draft_tok_;
    bool      draft_loaded_ = false;
    static constexpr int SPEC_N = 4; // tokens proposés par le draft par étape

    // Formate les messages dans le template chat Llama 3 / Qwen3
    std::string format_prompt(const std::vector<Message>& messages) const;
};
