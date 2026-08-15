#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# compare_llama.sh — Comparatif ORACLE vs llama.cpp sur la MÊME machine
#
# 1. Lance oracle_bench (via run_bench.sh) → result_<model>.json
# 2. Lance llama-bench (llama.cpp) avec les MÊMES paramètres
# 3. Fusionne les deux dans benchmark/comparison.md (tableau prêt à coller
#    dans BENCHMARKS.md, section Results)
#
# Prérequis :
#   • llama.cpp compilé : le binaire `llama-bench` accessible via $LLAMA_BENCH
#     ou dans le PATH. (build llama.cpp : cmake -B build && cmake --build build)
#   • python3 (parsing JSON)
#
# USAGE :
#   LLAMA_BENCH=/path/to/llama-bench \
#   PROMPT=128 GEN=128 RUNS=5 THREADS=8 \
#   MODELS="models/qwen2.5-7b-instruct-q4km.gguf" \
#     ./benchmark/compare_llama.sh
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$ROOT/benchmark"
OUT="$BENCH_DIR/comparison.md"

PROMPT="${PROMPT:-128}"
GEN="${GEN:-128}"
RUNS="${RUNS:-5}"
THREADS="${THREADS:-8}"
MODELS="${MODELS:-models/qwen2.5-7b-instruct-q4km.gguf}"

# --- Localiser llama-bench --------------------------------------------------
LB="${LLAMA_BENCH:-$(command -v llama-bench || true)}"
if [ -z "$LB" ] || [ ! -x "$LB" ]; then
    echo "ERREUR : llama-bench introuvable."
    echo "  Compile llama.cpp puis :  export LLAMA_BENCH=/chemin/vers/llama-bench"
    exit 1
fi
echo "[compare] llama-bench : $LB"

# --- 1. ORACLE --------------------------------------------------------------
echo "[compare] ── ORACLE ──"
PROMPT="$PROMPT" GEN="$GEN" RUNS="$RUNS" THREADS="$THREADS" MODELS="$MODELS" \
    "$BENCH_DIR/run_bench.sh"

# --- 2. + 3. llama.cpp puis fusion, par modèle ------------------------------
{
    echo "# ORACLE vs llama.cpp"
    echo
    echo "_Même machine, même modèle, même quantization, prompt=$PROMPT, gen=$GEN, threads=$THREADS, runs=$RUNS._"
    echo "_Généré le $(date '+%Y-%m-%d %H:%M')._"
    echo
} > "$OUT"

for m in $MODELS; do
    [ -f "$ROOT/$m" ] || { echo "[warn] modèle absent : $m"; continue; }
    name="$(basename "$m" | sed 's/\.[^.]*$//')"
    oracle_json="$BENCH_DIR/result_$name.json"
    llama_json="$BENCH_DIR/llama_$name.json"

    echo "[compare] ── llama.cpp : $m ──"
    # -o json : sortie machine ; -p prefill, -n génération, -r répétitions
    ( cd "$ROOT" && "$LB" -m "$m" -p "$PROMPT" -n "$GEN" -t "$THREADS" -r "$RUNS" -o json ) \
        > "$llama_json"

    python3 - "$name" "$m" "$oracle_json" "$llama_json" "$OUT" <<'PY'
import json, sys, os
name, model, oracle_path, llama_path, out_path = sys.argv[1:6]

def fmt(x):
    return f"{x:.1f}" if isinstance(x, (int, float)) else "—"

# ORACLE
o_pre = o_gen = o_rss = None
if os.path.exists(oracle_path):
    with open(oracle_path, encoding="utf-8") as f:
        o = json.load(f)
    o_pre = o["prefill_tok_s"]["median"]
    o_gen = o["generation_tok_s"]["median"]
    o_rss = o.get("peak_rss_mb")

# llama-bench : liste d'objets. pp = prefill (n_gen==0), tg = génération (n_prompt==0)
l_pre = l_gen = None
with open(llama_path, encoding="utf-8") as f:
    rows = json.load(f)
for r in rows:
    npr = int(r.get("n_prompt", 0)); ngn = int(r.get("n_gen", 0))
    ts  = float(r.get("avg_ts", r.get("t/s", 0)))
    if ngn == 0 and npr > 0:  l_pre = ts   # pp<N>
    if npr == 0 and ngn > 0:  l_gen = ts   # tg<N>

with open(out_path, "a", encoding="utf-8") as f:
    f.write(f"## {os.path.basename(model)}\n\n")
    f.write("| Engine | Prefill (tok/s) | Generation (tok/s) | Peak RSS |\n")
    f.write("|---|---|---|---|\n")
    f.write(f"| ORACLE | {fmt(o_pre)} | {fmt(o_gen)} | {fmt(o_rss)} MB |\n")
    f.write(f"| llama.cpp | {fmt(l_pre)} | {fmt(l_gen)} | (voir llama-bench) |\n\n")
    if o_gen and l_gen:
        ratio = o_gen / l_gen
        f.write(f"_Génération : ORACLE = **{ratio:.2f}×** llama.cpp._\n\n")
PY
done

echo "[compare] Comparatif écrit : $OUT"
cat "$OUT"
