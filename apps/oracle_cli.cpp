// ─────────────────────────────────────────────────────────────────────────
// oracle_cli.cpp — CLI minimal du moteur d'inférence ORACLE
//
//   oracle -m model.gguf -p "your prompt" [options]
//
// Options :
//   -m <path>       Modèle GGUF (obligatoire)
//   -p <text>       Prompt utilisateur (obligatoire)
//   -n <N>          Tokens à générer          (défaut 256)
//   --temp <f>      Température                (défaut 0.7)
//   --top-p <f>     Nucleus top-p             (défaut 0.9)
//   --system <text> Message système
//   --draft <path>  Modèle draft (speculative decoding)
//   --no-stream     Ne pas streamer (affiche la réponse en un bloc)
// ─────────────────────────────────────────────────────────────────────────
#include "core/engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static void usage(const char* exe) {
    std::fprintf(stderr,
        "Usage: %s -m <model.gguf> -p <prompt> [options]\n"
        "  -n <N>          tokens to generate      (default 256)\n"
        "  --temp <f>      temperature             (default 0.7)\n"
        "  --top-p <f>     nucleus top-p           (default 0.9)\n"
        "  --system <text> system message\n"
        "  --draft <path>  draft model (speculative decoding)\n"
        "  --no-stream     print the answer in one block\n", exe);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string model, prompt, system, draft;
    GenParams params;
    bool stream = true;

    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto val = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", n); std::exit(2); }
            return argv[++i];
        };
        if      (k == "-m")          model  = val("-m");
        else if (k == "-p")          prompt = val("-p");
        else if (k == "-n")          params.n_predict   = std::atoi(val("-n"));
        else if (k == "--temp")      params.temperature = (float)std::atof(val("--temp"));
        else if (k == "--top-p")     params.top_p       = (float)std::atof(val("--top-p"));
        else if (k == "--system")    system = val("--system");
        else if (k == "--draft")     draft  = val("--draft");
        else if (k == "--no-stream") stream = false;
        else if (k == "-h" || k == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown option: %s\n", k.c_str()); usage(argv[0]); return 2; }
    }
    if (model.empty() || prompt.empty()) { usage(argv[0]); return 2; }

    Engine engine;
    if (!engine.load(model, params)) {
        std::fprintf(stderr, "error: failed to load model '%s'\n", model.c_str());
        return 1;
    }
    if (!draft.empty() && !engine.load_draft(draft))
        std::fprintf(stderr, "warning: failed to load draft model '%s' (continuing without)\n", draft.c_str());

    std::vector<Message> messages;
    if (!system.empty()) messages.push_back({"system", system});
    messages.push_back({"user", prompt});

    std::string answer = engine.generate(
        messages, params,
        stream ? std::function<void(const std::string&)>(
                     [](const std::string& piece) { std::fputs(piece.c_str(), stdout); std::fflush(stdout); })
               : nullptr);

    if (!stream) std::fputs(answer.c_str(), stdout);
    std::fputc('\n', stdout);
    return 0;
}
