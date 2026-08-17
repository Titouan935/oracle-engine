// ─────────────────────────────────────────────────────────────────────────
// bench_engine.cpp — Harnais de benchmark reproductible du moteur ORACLE
//
// Mesure, séparément et de façon déterministe :
//   • Prefill   : débit (tok/s) du traitement batché du prompt (forward_prefill)
//   • Génération: débit (tok/s) de la décode autorégressive (greedy/argmax)
//   • Peak RSS  : mémoire résidente maximale du process
//
// Méthodo (défendable en public) :
//   • Génération GREEDY (argmax) → sortie déterministe, indépendante du sampler
//   • Prompt fixe, tuilé jusqu'à la longueur demandée → reproductible
//   • R runs, MÉDIANE reportée, 1er run jeté (warmup) en plus de model.warmup()
//   • KV cache remis à zéro (reset) avant chaque run
//   • Nombre de threads fixé (-t), imprimé dans le rapport
//   • Checksum des tokens générés imprimé → deux runs identiques = même hash
//
// Sorties : humain (stdout) + fragment Markdown (--md) + JSON (--json).
//
// Usage :
//   oracle_bench -m models/qwen2.5-7b-instruct-q4km.gguf \
//                -p 128 -g 128 -r 5 -t 8 \
//                --md bench.md --json bench.json
// ─────────────────────────────────────────────────────────────────────────
#include "core/gguf_parser.hpp"
#include "core/tokenizer.hpp"
#include "core/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifndef BENCH_CXX_FLAGS
#define BENCH_CXX_FLAGS "unknown"
#endif
#ifndef BENCH_COMPILER
#define BENCH_COMPILER "unknown"
#endif

// ── Peak RSS en octets, portable ──────────────────────────────────────────
static uint64_t peak_rss_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (uint64_t)pmc.PeakWorkingSetSize;
    return 0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
        return (uint64_t)ru.ru_maxrss;        // macOS : octets
#else
        return (uint64_t)ru.ru_maxrss * 1024; // Linux : kilo-octets
#endif
    }
    return 0;
#endif
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static int32_t argmax(const float* v, int n) {
    int32_t best = 0;
    float bv = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

// Prompt de référence (texte neutre, anglais) — tuilé jusqu'à la longueur voulue.
static const char* REF_PROMPT =
    "The quick brown fox jumps over the lazy dog. In the beginning the engine "
    "reads the weights, decodes each quantized block, and streams one token at "
    "a time. Performance lives in the matmul kernel where most of the wall "
    "clock time hides, and every architectural decision is measured against it. ";

struct Args {
    std::string model;
    int   n_prompt = 128;
    int   n_gen    = 128;
    int   runs     = 5;   // dont 1 jeté (warmup)
    int   threads  = 0;   // 0 = laisser OpenMP décider
    std::string md_out;
    std::string json_out;
    std::string draft;    // modèle draft → mesure speculative decoding
    int   spec_n   = 4;   // tokens proposés par le draft par étape
};

static void usage(const char* exe) {
    std::fprintf(stderr,
        "Usage : %s -m <model.gguf> [options]\n"
        "  -m <path>     Modèle GGUF (obligatoire)\n"
        "  -p <N>        Tokens de prompt à préfiller     (défaut 128)\n"
        "  -g <N>        Tokens à générer                 (défaut 128)\n"
        "  -r <N>        Runs (1er jeté en warmup)         (défaut 5)\n"
        "  -t <N>        Threads (0 = auto OpenMP)         (défaut 0)\n"
        "  --md <path>   Écrit un fragment Markdown\n"
        "  --json <path> Écrit un rapport JSON\n"
        "  --draft <p>   Modèle draft → mesure le speculative decoding\n"
        "  --spec-n <N>  Tokens proposés par le draft/étape   (défaut 4)\n", exe);
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "manque valeur après %s\n", name); return nullptr; }
            return argv[++i];
        };
        if      (k == "-m") { const char* v = next("-m"); if (!v) return false; a.model = v; }
        else if (k == "-p") { const char* v = next("-p"); if (!v) return false; a.n_prompt = std::atoi(v); }
        else if (k == "-g") { const char* v = next("-g"); if (!v) return false; a.n_gen = std::atoi(v); }
        else if (k == "-r") { const char* v = next("-r"); if (!v) return false; a.runs = std::atoi(v); }
        else if (k == "-t") { const char* v = next("-t"); if (!v) return false; a.threads = std::atoi(v); }
        else if (k == "--md")   { const char* v = next("--md");   if (!v) return false; a.md_out = v; }
        else if (k == "--json") { const char* v = next("--json"); if (!v) return false; a.json_out = v; }
        else if (k == "--draft"){ const char* v = next("--draft");if (!v) return false; a.draft = v; }
        else if (k == "--spec-n"){const char* v = next("--spec-n");if (!v) return false; a.spec_n = std::atoi(v); }
        else if (k == "-h" || k == "--help") { usage(argv[0]); return false; }
        else { std::fprintf(stderr, "option inconnue : %s\n", k.c_str()); return false; }
    }
    if (a.model.empty())    { std::fprintf(stderr, "ERREUR : -m <model.gguf> requis\n\n"); usage(argv[0]); return false; }
    if (a.n_prompt < 2)     a.n_prompt = 2;
    if (a.n_gen    < 1)     a.n_gen    = 1;
    if (a.runs     < 1)     a.runs     = 1;
    return true;
}

// Nom de fichier seul (pour l'affichage), sans le chemin.
static std::string basename_of(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

#ifdef _OPENMP
    if (args.threads > 0) omp_set_num_threads(args.threads);
    int threads_used = omp_get_max_threads();
#else
    int threads_used = 1;
#endif

    std::printf("═══════════════════════════════════════════════════════════\n");
    std::printf("  ORACLE — benchmark moteur (prefill / génération / RSS)\n");
    std::printf("═══════════════════════════════════════════════════════════\n");

    // ── Chargement ────────────────────────────────────────────────────────
    Model model;
    if (!model.load(args.model)) {
        std::fprintf(stderr, "ERREUR : chargement modèle échoué (%s)\n", args.model.c_str());
        return 1;
    }
    const auto& cfg = model.cfg;

    Tokenizer tok;
    if (!tok.load(model.gguf())) {
        std::fprintf(stderr, "ERREUR : chargement tokenizer échoué\n");
        return 1;
    }

    // Validation contexte : prompt + génération doivent tenir dans ctx_len.
    if ((uint32_t)(args.n_prompt + args.n_gen) > cfg.ctx_len) {
        int room = (int)cfg.ctx_len - args.n_gen;
        if (room < 2) { room = 2; args.n_gen = (int)cfg.ctx_len - room; }
        std::fprintf(stderr,
            "[warn] prompt(%d)+gen(%d) > ctx_len(%u) : prompt réduit à %d\n",
            args.n_prompt, args.n_gen, cfg.ctx_len, room);
        args.n_prompt = room;
    }

    // ── Construction du prompt : tuilage jusqu'à n_prompt tokens ───────────
    std::vector<int32_t> base = tok.encode(REF_PROMPT, /*add_bos=*/true);
    if (base.empty()) { std::fprintf(stderr, "ERREUR : tokenisation vide\n"); return 1; }
    std::vector<int32_t> prompt;
    prompt.reserve(args.n_prompt);
    while ((int)prompt.size() < args.n_prompt)
        prompt.push_back(base[prompt.size() % base.size()]);
    prompt.resize(args.n_prompt);

    const int prefill_count = args.n_prompt - 1;   // dernier token gardé pour la 1re passe logits

    std::printf("Modèle    : %s\n", basename_of(args.model).c_str());
    std::printf("Config    : %u couches, %u/%u têtes Q/KV, embed=%u, ffn=%u, vocab=%u, ctx=%u\n",
                cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.embed_dim,
                cfg.ffn_hidden, cfg.n_vocab, cfg.ctx_len);
    std::printf("Threads   : %d\n", threads_used);
    std::printf("Compilo   : %s | flags: %s\n", BENCH_COMPILER, BENCH_CXX_FLAGS);
    std::printf("Prefill   : %d tokens | Génération : %d tokens | Runs : %d (1 jeté)\n\n",
                prefill_count, args.n_gen, args.runs);

    std::printf("Warmup pages RAM...\n");
    model.warmup();

    // ── Boucle de runs ────────────────────────────────────────────────────
    std::vector<double> prefill_tps, gen_tps;
    uint64_t gen_checksum = 0;   // du dernier run mesuré (déterministe)

    for (int run = 0; run < args.runs; run++) {
        model.reset();

        // Prefill batché (positions 0..prefill_count-1)
        auto p0 = std::chrono::high_resolution_clock::now();
        model.forward_prefill(prompt.data(), prefill_count, 0);
        auto p1 = std::chrono::high_resolution_clock::now();
        double dt_prefill = std::chrono::duration<double>(p1 - p0).count();

        // Génération greedy à partir du dernier token du prompt
        int32_t cur = prompt[prefill_count];   // = prompt.back()
        int pos = prefill_count;
        uint64_t checksum = 1469598103934665603ULL; // FNV-1a offset

        auto g0 = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < args.n_gen; step++) {
            const float* logits = model.forward(cur, pos);
            pos++;
            cur = argmax(logits, (int)cfg.n_vocab);
            checksum ^= (uint64_t)(uint32_t)cur;
            checksum *= 1099511628211ULL;       // FNV-1a prime
        }
        auto g1 = std::chrono::high_resolution_clock::now();
        double dt_gen = std::chrono::duration<double>(g1 - g0).count();

        double ptps = dt_prefill > 0 ? prefill_count / dt_prefill : 0.0;
        double gtps = dt_gen     > 0 ? args.n_gen    / dt_gen     : 0.0;

        bool warm = (run == 0 && args.runs > 1);
        std::printf("  run %d/%d%s : prefill %.1f tok/s (%.3fs) | gen %.1f tok/s (%.3fs)\n",
                    run + 1, args.runs, warm ? " [warmup, jeté]" : "",
                    ptps, dt_prefill, gtps, dt_gen);

        if (!warm) {
            prefill_tps.push_back(ptps);
            gen_tps.push_back(gtps);
            gen_checksum = checksum;
        }
    }

    // ── Résultats ─────────────────────────────────────────────────────────
    double p_med = median(prefill_tps);
    double g_med = median(gen_tps);
    double p_min = *std::min_element(prefill_tps.begin(), prefill_tps.end());
    double p_max = *std::max_element(prefill_tps.begin(), prefill_tps.end());
    double g_min = *std::min_element(gen_tps.begin(), gen_tps.end());
    double g_max = *std::max_element(gen_tps.begin(), gen_tps.end());
    uint64_t rss = peak_rss_bytes();
    double rss_mb = rss / (1024.0 * 1024.0);

    std::printf("\n───────────────────────────────────────────────────────────\n");
    std::printf("RÉSULTATS (médiane sur %d runs mesurés)\n", (int)gen_tps.size());
    std::printf("  Prefill    : %.1f tok/s   [min %.1f, max %.1f]\n", p_med, p_min, p_max);
    std::printf("  Génération : %.1f tok/s   [min %.1f, max %.1f]\n", g_med, g_min, g_max);
    std::printf("  Peak RSS   : %.0f MB\n", rss_mb);
    std::printf("  Checksum   : %016llx  (déterministe : identique entre runs)\n",
                (unsigned long long)gen_checksum);
    std::printf("───────────────────────────────────────────────────────────\n");

    // ── Speculative decoding (si --draft) ─────────────────────────────────
    // Le spec-decode est EXACT : il produit exactement les mêmes tokens que le
    // modèle principal seul. Donc le checksum doit être identique à la
    // génération normale, quel que soit le draft (invariant de correction).
    // Le taux d'acceptation et les forwards main/token mesurent le gain.
    if (!args.draft.empty()) {
        std::printf("\nSpeculative decoding — draft : %s\n", basename_of(args.draft).c_str());
        Model draft;
        if (!draft.load(args.draft)) {
            std::fprintf(stderr, "[warn] chargement draft échoué (%s)\n", args.draft.c_str());
        } else if (draft.cfg.n_vocab != cfg.n_vocab) {
            std::fprintf(stderr, "[warn] vocab draft (%u) != main (%u) — spec-decode impossible\n",
                         draft.cfg.n_vocab, cfg.n_vocab);
        } else {
            draft.warmup();
            const int SPEC_N = std::min(std::max(args.spec_n, 1), 64);
            std::vector<float> blog((size_t)(SPEC_N + 1) * cfg.n_vocab); // logits batchés de vérif
            double spec_tps = 0.0; uint64_t spec_checksum = 0;
            long m_fwd = 0, gen = 0, acc = 0, prop = 0;

            for (int run = 0; run < 2; run++) {           // run 0 = warmup, run 1 = mesuré
                model.reset(); draft.reset();
                model.forward_prefill(prompt.data(), prefill_count, 0);
                draft.forward_prefill(prompt.data(), prefill_count, 0);
                int pos = prefill_count;
                int32_t next = prompt[prefill_count];
                long lm_fwd = 0, lgen = 0, lacc = 0, lprop = 0;
                uint64_t csum = 1469598103934665603ULL;
                const int max_ctx = (int)cfg.ctx_len;

                auto s0 = std::chrono::high_resolution_clock::now();
                while (lgen < args.n_gen && pos < max_ctx - 1) {
                    // a) draft propose SPEC_N tokens greedily
                    int32_t dtok[64]; int ndt = 0;
                    { int dp = pos; int32_t dt = next;
                      for (int k = 0; k < SPEC_N && dp < (int)draft.cfg.ctx_len - 1; k++) {
                          const float* dl = draft.forward(dt, dp++);
                          if (!dl) break;
                          int32_t b = argmax(dl, (int)draft.cfg.n_vocab);
                          dtok[ndt++] = b; dt = b;
                      } }
                    if (ndt == 0) break;
                    lprop += ndt;

                    // b) main vérifie les ndt tokens en UNE passe batchée.
                    //    Entrée = [next, d0, ..., d_{ndt-1}] aux positions
                    //    pos..pos+ndt ; L[j] prédit le token après vin[j].
                    //    Les poids (surtout le LM head) ne sont lus QU'UNE fois.
                    const int V = (int)cfg.n_vocab;
                    int nver = ndt + 1;
                    if (pos + nver > max_ctx) nver = max_ctx - pos;
                    if (nver < 1) break;
                    int32_t vin[65];
                    vin[0] = next;
                    for (int k = 0; k < nver - 1; k++) vin[k + 1] = dtok[k];
                    model.forward_batch(vin, nver, pos, blog.data()); lm_fwd++;   // 1 passe = poids lus 1×

                    int a = 0; bool brk = false;
                    for (int k = 0; k < ndt && k < nver; k++) {
                        int32_t mtok = argmax(blog.data() + (size_t)k * V, V);
                        lgen++; csum ^= (uint64_t)(uint32_t)mtok; csum *= 1099511628211ULL;
                        if (mtok != dtok[k]) { next = mtok; a = k; brk = true; break; }
                        lacc++; a = k + 1; next = dtok[k];
                        if (lgen >= args.n_gen) { brk = true; break; }
                    }
                    pos += a + 1;   // next_old + a tokens acceptés consommés

                    if (brk) continue;   // rejet partiel / n_gen atteint : pas de bonus

                    // c) tous acceptés → bonus = argmax(L[ndt]) + resync KV draft
                    if (lgen < args.n_gen && ndt < nver && pos < max_ctx - 1) {
                        int32_t b = argmax(blog.data() + (size_t)ndt * V, V);
                        lgen++; csum ^= (uint64_t)(uint32_t)b; csum *= 1099511628211ULL;
                        next = b;
                        draft.forward(dtok[ndt - 1], pos - 1, false);   // comble la position manquante du draft
                    }
                }
                double dt = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - s0).count();
                if (run == 1) {
                    spec_tps = (lgen > 0 && dt > 0) ? lgen / dt : 0.0;
                    spec_checksum = csum; m_fwd = lm_fwd; gen = lgen; acc = lacc; prop = lprop;
                }
            }

            double accept  = prop > 0 ? 100.0 * acc / prop : 0.0;
            double fwd_tok = gen  > 0 ? (double)m_fwd / gen : 0.0;
            double speedup = g_med > 0 ? spec_tps / g_med : 0.0;
            bool   exact   = (spec_checksum == gen_checksum);
            std::printf("  Génération (spec) : %.1f tok/s   (×%.2f vs greedy simple %.1f tok/s)\n",
                        spec_tps, speedup, g_med);
            std::printf("  Acceptation       : %.1f%%   (%ld/%ld tokens draft acceptés, SPEC_N=%d)\n",
                        accept, acc, prop, SPEC_N);
            std::printf("  Passes main/tok   : %.2f   (poids lus/token ; 1.00 = pas de gain, plus bas = mieux)\n", fwd_tok);
            std::printf("  Sortie exacte     : %s  (checksum spec %016llx vs greedy %016llx)\n",
                        exact ? "OUI ✓" : "NON ✗ (BUG)",
                        (unsigned long long)spec_checksum, (unsigned long long)gen_checksum);
            std::printf("───────────────────────────────────────────────────────────\n");
        }
    }

    // ── Fragment Markdown (pour BENCHMARKS.md) ────────────────────────────
    if (!args.md_out.empty()) {
        FILE* f = std::fopen(args.md_out.c_str(), "w");
        if (f) {
            std::fprintf(f, "| Engine | Model | Prefill (tok/s) | Generation (tok/s) | Peak RSS | Threads |\n");
            std::fprintf(f, "|---|---|---|---|---|---|\n");
            std::fprintf(f, "| ORACLE | %s | %.1f | %.1f | %.0f MB | %d |\n",
                         basename_of(args.model).c_str(), p_med, g_med, rss_mb, threads_used);
            std::fprintf(f, "\n_Median of %d runs (1 warmup discarded). Greedy decode, prompt=%d tok, gen=%d tok. "
                            "Compiler: %s, flags: `%s`. Deterministic checksum: `%016llx`._\n",
                         (int)gen_tps.size(), prefill_count, args.n_gen,
                         BENCH_COMPILER, BENCH_CXX_FLAGS, (unsigned long long)gen_checksum);
            std::fclose(f);
            std::printf("→ Markdown écrit : %s\n", args.md_out.c_str());
        } else {
            std::fprintf(stderr, "[warn] impossible d'écrire %s\n", args.md_out.c_str());
        }
    }

    // ── Rapport JSON ──────────────────────────────────────────────────────
    if (!args.json_out.empty()) {
        FILE* f = std::fopen(args.json_out.c_str(), "w");
        if (f) {
            std::fprintf(f,
                "{\n"
                "  \"engine\": \"ORACLE\",\n"
                "  \"model\": \"%s\",\n"
                "  \"compiler\": \"%s\",\n"
                "  \"flags\": \"%s\",\n"
                "  \"threads\": %d,\n"
                "  \"prompt_tokens\": %d,\n"
                "  \"gen_tokens\": %d,\n"
                "  \"runs_measured\": %d,\n"
                "  \"prefill_tok_s\": { \"median\": %.3f, \"min\": %.3f, \"max\": %.3f },\n"
                "  \"generation_tok_s\": { \"median\": %.3f, \"min\": %.3f, \"max\": %.3f },\n"
                "  \"peak_rss_mb\": %.1f,\n"
                "  \"gen_checksum\": \"%016llx\"\n"
                "}\n",
                basename_of(args.model).c_str(), BENCH_COMPILER, BENCH_CXX_FLAGS,
                threads_used, prefill_count, args.n_gen, (int)gen_tps.size(),
                p_med, p_min, p_max, g_med, g_min, g_max, rss_mb,
                (unsigned long long)gen_checksum);
            std::fclose(f);
            std::printf("→ JSON écrit     : %s\n", args.json_out.c_str());
        } else {
            std::fprintf(stderr, "[warn] impossible d'écrire %s\n", args.json_out.c_str());
        }
    }

    return 0;
}
