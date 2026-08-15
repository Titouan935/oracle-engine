#include "tokenizer.hpp"
#include "bug/logger.hpp"
#include <algorithm>
#include <climits>
#include <sstream>
#include <cassert>

// ══════════════════════════════════════════════════════════════════════════
//  Tables byte ↔ unicode (style GPT-2 — identique à bytes_to_unicode())
//
//  Les bytes 33-126, 161-172, 174-255 se mappent vers eux-mêmes.
//  Les bytes manquants (0-32, 127-160, 173) se mappent vers 256, 257, ...
//  → byte 32 (espace) = U+0120 = 'Ġ'
// ══════════════════════════════════════════════════════════════════════════
void Tokenizer::build_byte_tables() {
    // 1. Collecte les bytes qui se mappent vers eux-mêmes
    std::vector<int> bs, cs;
    for (int b = 33; b <= 126; b++)  { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xA1; b <= 0xAC; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 0xAE; b <= 0xFF; b++) { bs.push_back(b); cs.push_back(b); }

    // 2. Les bytes absents reçoivent 256+n
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }

    // 3. Remplit les tables
    for (int i = 0; i < (int)bs.size(); i++) {
        b2u_[bs[i]] = (uint32_t)cs[i];
        if (cs[i] < 512) u2b_[cs[i]] = (uint8_t)bs[i];
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  Chargement depuis GGUF
// ══════════════════════════════════════════════════════════════════════════
bool Tokenizer::load(const GGUFParser& gguf) {
    build_byte_tables();

    // Vocabulaire
    const auto& toks = gguf.meta_arr_str("tokenizer.ggml.tokens");
    if (toks.empty()) {
        LOG_ERR("Tokenizer", "tokenizer.ggml.tokens absent du GGUF");
        return false;
    }
    tokens_ = toks;

    // Scores et types (optionnels)
    const auto& sc_meta = gguf.meta("tokenizer.ggml.scores");
    if (sc_meta.kind == MetaValue::ARR_F32) scores_ = sc_meta.arr_f32;

    const auto& ty_meta = gguf.meta("tokenizer.ggml.token_type");
    if (ty_meta.kind == MetaValue::ARR_I32) types_ = ty_meta.arr_i32;

    // Index inverse
    tok2id_.reserve(tokens_.size());
    for (int32_t i = 0; i < (int32_t)tokens_.size(); i++)
        tok2id_[tokens_[i]] = i;

    // Règles de fusion BPE
    const auto& mgs = gguf.meta_arr_str("tokenizer.ggml.merges");
    merges_.reserve(mgs.size());
    for (int i = 0; i < (int)mgs.size(); i++)
        merges_[mgs[i]] = i;   // clé : "piece1 piece2"

    // Tokens spéciaux
    bos_id_ = (int32_t)gguf.meta_u32("tokenizer.ggml.bos_token_id", 1);
    eos_id_ = (int32_t)gguf.meta_u32("tokenizer.ggml.eos_token_id", 2);

    LOG_INFO("Tokenizer", "Chargé — vocab=" + std::to_string(tokens_.size())
             + " merges=" + std::to_string(merges_.size())
             + " bos=" + std::to_string(bos_id_)
             + " eos=" + std::to_string(eos_id_));
    return true;
}

// ══════════════════════════════════════════════════════════════════════════
//  Helpers UTF-8
// ══════════════════════════════════════════════════════════════════════════
uint32_t Tokenizer::next_codepoint(const char* p, int& len) {
    unsigned char c = (unsigned char)*p;
    if (c < 0x80)       { len = 1; return c; }
    if (c < 0xE0)       { len = 2; return ((c & 0x1F) << 6)  | ((unsigned char)p[1] & 0x3F); }
    if (c < 0xF0)       { len = 3; return ((c & 0x0F) << 12) | ((unsigned char)p[1] & 0x3F) << 6  | ((unsigned char)p[2] & 0x3F); }
    /* 4-byte */         { len = 4; return ((c & 0x07) << 18) | ((unsigned char)p[1] & 0x3F) << 12 | ((unsigned char)p[2] & 0x3F) << 6 | ((unsigned char)p[3] & 0x3F); }
}

std::string Tokenizer::cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6)  & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

std::vector<std::string> Tokenizer::split_codepoints(const std::string& s) {
    std::vector<std::string> out;
    const char* p = s.c_str();
    const char* end = p + s.size();
    while (p < end) {
        int len = 1;
        uint32_t cp = next_codepoint(p, len);
        out.push_back(cp_to_utf8(cp));
        p += len;
    }
    return out;
}

// ══════════════════════════════════════════════════════════════════════════
//  Encodage/décodage GPT-2
// ══════════════════════════════════════════════════════════════════════════

// Convertit chaque byte de `raw` vers le code point GPT-2 correspondant
std::string Tokenizer::to_gpt2(const std::string& raw) const {
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char b : raw)
        out += cp_to_utf8(b2u_[b]);
    return out;
}

// Convertit chaque code point GPT-2 (dans `gpt2`) vers le byte d'origine
std::string Tokenizer::from_gpt2(const std::string& gpt2) const {
    std::string out;
    const char* p   = gpt2.c_str();
    const char* end = p + gpt2.size();
    while (p < end) {
        int len; uint32_t cp = next_codepoint(p, len);
        if (cp < 512)
            out += (char)u2b_[cp];
        else
            out += cp_to_utf8(cp);   // code point hors table → garde tel quel
        p += len;
    }
    return out;
}

// ══════════════════════════════════════════════════════════════════════════
//  Algorithme BPE (Byte Pair Encoding)
// ══════════════════════════════════════════════════════════════════════════
std::vector<int32_t> Tokenizer::bpe(const std::string& gpt2_word) const {
    // 1. Découpe en code points individuels
    std::vector<std::string> pieces = split_codepoints(gpt2_word);
    if (pieces.empty()) return {};

    // 2. Fusions itératives : cherche la fusion de rang le plus bas
    while (pieces.size() > 1) {
        int  best_rank = INT_MAX;
        int  best_i    = -1;

        for (int i = 0; i < (int)pieces.size() - 1; i++) {
            std::string key = pieces[i] + " " + pieces[i + 1];
            auto it = merges_.find(key);
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i    = i;
            }
        }

        if (best_i == -1) break;  // plus de fusions possibles

        // Applique la fusion
        pieces[best_i] += pieces[best_i + 1];
        pieces.erase(pieces.begin() + best_i + 1);
    }

    // 3. Convertit chaque pièce en ID
    std::vector<int32_t> ids;
    ids.reserve(pieces.size());
    for (const auto& p : pieces) {
        auto it = tok2id_.find(p);
        if (it != tok2id_.end()) {
            ids.push_back(it->second);
        } else {
            // Pièce inconnue → encode byte par byte (tokens <0xXX>)
            for (unsigned char b : from_gpt2(p)) {
                char buf[8];
                snprintf(buf, sizeof(buf), "<0x%02X>", b);
                auto jt = tok2id_.find(buf);
                if (jt != tok2id_.end()) ids.push_back(jt->second);
                else                      ids.push_back(0);  // UNK
            }
        }
    }
    return ids;
}

// ══════════════════════════════════════════════════════════════════════════
//  Pré-tokenisation (style GPT-2 — espace attaché au mot suivant)
//
//  "Bonjour monde" → ["Bonjour", " monde"]
//  puis chaque " monde" sera to_gpt2-encodé → "Ġmonde"
// ══════════════════════════════════════════════════════════════════════════
std::vector<std::string> Tokenizer::pretok(const std::string& text) const {
    std::vector<std::string> words;
    std::string cur;

    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];

        // Détecte un espace (ASCII ou début d'un mot après espace)
        if (c == ' ' || c == '\t') {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
            // Attache l'espace au mot suivant
            cur += (char)c;
            i++;
        } else if (c == '\n' || c == '\r') {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
            cur += (char)c;
            words.push_back(cur); cur.clear();
            i++;
        } else {
            // Avance d'un code point UTF-8 complet
            int len = 1;
            if      (c < 0x80) len = 1;
            else if (c < 0xE0) len = 2;
            else if (c < 0xF0) len = 3;
            else               len = 4;
            for (int j = 0; j < len && i < text.size(); j++, i++)
                cur += text[i];
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

// ══════════════════════════════════════════════════════════════════════════
//  encode : texte → IDs de tokens
// ══════════════════════════════════════════════════════════════════════════
std::vector<int32_t> Tokenizer::encode(const std::string& text,
                                        bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos) ids.push_back(bos_id_);

    auto words = pretok(text);
    for (const auto& w : words) {
        // Vérifie d'abord si le mot entier est un token spécial connu
        auto it = tok2id_.find(w);
        if (it != tok2id_.end() &&
            !types_.empty() &&
            (int32_t)types_.size() > it->second &&
            types_[it->second] >= 3) {
            ids.push_back(it->second);
            continue;
        }
        // Sinon : encodage GPT-2 + BPE
        std::string g = to_gpt2(w);
        auto word_ids = bpe(g);
        ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }
    return ids;
}

// ══════════════════════════════════════════════════════════════════════════
//  decode_one : ID → texte brut (bytes)
// ══════════════════════════════════════════════════════════════════════════
std::string Tokenizer::decode_one(int32_t id) const {
    if (id < 0 || id >= (int32_t)tokens_.size()) return "";
    const std::string& tok = tokens_[id];

    // Tokens de type BYTE (<0xXX>) → byte direct
    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x') {
        unsigned int b = 0;
        if (sscanf(tok.c_str() + 3, "%02x>", &b) == 1)
            return std::string(1, (char)b);
    }

    // Tokens de contrôle (spéciaux) → retourner tel quel
    if (!types_.empty() && id < (int32_t)types_.size() && types_[id] >= 3)
        return "";  // ne pas afficher les tokens spéciaux dans le texte

    // Tokens normaux : décoder depuis GPT-2
    return from_gpt2(tok);
}

// ══════════════════════════════════════════════════════════════════════════
//  decode : liste d'IDs → texte
// ══════════════════════════════════════════════════════════════════════════
std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += decode_one(id);
    return out;
}

// ══════════════════════════════════════════════════════════════════════════
//  special_id : cherche l'ID d'un token spécial par son nom exact
// ══════════════════════════════════════════════════════════════════════════
int32_t Tokenizer::special_id(const std::string& name) const {
    auto it = tok2id_.find(name);
    return it != tok2id_.end() ? it->second : -1;
}
