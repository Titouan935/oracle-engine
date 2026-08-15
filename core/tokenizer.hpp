#pragma once
#include "gguf_parser.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────
// Tokenizer BPE — chargé directement depuis les métadonnées GGUF
//
// Usage :
//   Tokenizer tok;
//   tok.load(parser);
//   auto ids = tok.encode("Bonjour ORACLE !");   // → [token_ids...]
//   auto txt = tok.decode(ids);                  // → "Bonjour ORACLE !"
// ─────────────────────────────────────────────────────────────────────────
class Tokenizer {
public:
    bool load(const GGUFParser& gguf);
    bool is_loaded() const { return !tokens_.empty(); }

    // Texte → IDs (add_bos ajoute <|begin_of_text|> au début)
    std::vector<int32_t> encode(const std::string& text,
                                bool add_bos = true) const;

    // IDs → texte
    std::string decode(const std::vector<int32_t>& ids) const;

    // Décode un seul token → octets UTF-8 bruts
    std::string decode_one(int32_t id) const;

    // Cherche l'ID d'un token spécial par son nom exact
    int32_t special_id(const std::string& name) const;

    int32_t  bos_id()     const { return bos_id_; }
    int32_t  eos_id()     const { return eos_id_; }
    uint32_t vocab_size() const { return (uint32_t)tokens_.size(); }

private:
    std::vector<std::string>              tokens_;   // id → chaîne brute
    std::vector<float>                    scores_;
    std::vector<int32_t>                  types_;
    std::unordered_map<std::string,int32_t> tok2id_;
    std::unordered_map<std::string,int>     merges_; // "a b" → rang

    int32_t bos_id_ = 128000;
    int32_t eos_id_ = 128009;

    // Tables de correspondance byte ↔ point de code unicode (style GPT-2)
    uint32_t b2u_[256] = {};   // byte → code point unicode
    uint8_t  u2b_[512] = {};   // code point unicode → byte (0..~323)

    void build_byte_tables();

    // Convertit du texte brut (bytes) vers l'encodage GPT-2 unicode
    // " hello" → "Ġhello" (Ġ = U+0120, encode en UTF-8 : C4 A0)
    std::string to_gpt2(const std::string& raw) const;

    // Convertit une chaîne GPT-2 encodée → bytes bruts
    // "Ġhello" → " hello"
    std::string from_gpt2(const std::string& gpt2) const;

    // Applique les fusions BPE à un mot déjà encodé GPT-2
    // Retourne la liste d'IDs de tokens correspondants
    std::vector<int32_t> bpe(const std::string& gpt2_word) const;

    // Découpe le texte en mots avec gestion des espaces (style GPT-2)
    // "Bonjour monde" → ["Bonjour", "Ġmonde"]
    std::vector<std::string> pretok(const std::string& text) const;

    // Parse le prochain code point UTF-8 à partir de p, retourne sa longueur
    static uint32_t next_codepoint(const char* p, int& len);

    // Convertit un code point unicode → chaîne UTF-8
    static std::string cp_to_utf8(uint32_t cp);

    // Découpe une chaîne UTF-8 en liste de code points (chacun en UTF-8)
    static std::vector<std::string> split_codepoints(const std::string& s);
};
