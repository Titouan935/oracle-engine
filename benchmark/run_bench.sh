#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# run_bench.sh — Benchmark reproductible d'ORACLE (macOS / Linux)
#
# Construit oracle_bench via CMake puis le lance sur une liste de modèles, et
# assemble les fragments Markdown dans benchmark/results.md.
#
# USAGE :
#   ./benchmark/run_bench.sh
#   PROMPT=128 GEN=128 RUNS=5 THREADS=8 \
#     MODELS="models/qwen2.5-7b-instruct-q4km.gguf" ./benchmark/run_bench.sh
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build"
EXE="$BUILD_DIR/oracle_bench"
BENCH_DIR="$ROOT/benchmark"
OUT_MD="$BENCH_DIR/results.md"

PROMPT="${PROMPT:-128}"
GEN="${GEN:-128}"
RUNS="${RUNS:-5}"
THREADS="${THREADS:-0}"
MODELS="${MODELS:-models/qwen2.5-7b-instruct-q4km.gguf models/llama-3.2-3b-q4k.gguf}"

# Convertit un chemin vers la forme native attendue par l'exe.
# Sur MSYS/Git Bash (Windows), l'exe est un binaire natif : CreateFileA/fopen
# exigent un chemin Windows (D:\...) — un chemin POSIX relatif échoue si le CWD
# n'est pas correctement traduit. cygpath -w règle ça. Sur un vrai Unix (Mac
# Pro), cygpath est absent → on renvoie le chemin absolu POSIX, qui marche.
native_path() {
    if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

# --- Build ------------------------------------------------------------------
echo "[bench] Configure + build oracle_bench..."
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD_DIR" --target oracle_bench -j >/dev/null
[ -x "$EXE" ] || { echo "ERREUR : oracle_bench non produit"; exit 1; }

# --- Runs -------------------------------------------------------------------
frags=""
for m in $MODELS; do
    if [ ! -f "$ROOT/$m" ]; then echo "[warn] modèle absent, ignoré : $m"; continue; fi
    name="$(basename "$m" | sed 's/\.[^.]*$//')"
    frag="$BENCH_DIR/_frag_$name.md"
    json="$BENCH_DIR/result_$name.json"

    echo "[bench] $m  (p=$PROMPT g=$GEN r=$RUNS t=$THREADS)"
    # Chemins natifs absolus → robustes quel que soit le CWD/traduction MSYS.
    m_native="$(native_path "$ROOT/$m")"
    frag_native="$(native_path "$frag")"
    json_native="$(native_path "$json")"
    targs=(-m "$m_native" -p "$PROMPT" -g "$GEN" -r "$RUNS" --md "$frag_native" --json "$json_native")
    [ "$THREADS" -gt 0 ] && targs+=(-t "$THREADS")

    ( cd "$ROOT" && "$EXE" "${targs[@]}" )
    [ -f "$frag" ] && { frags+="$(cat "$frag")"$'\n\n'; rm -f "$frag"; }
done

# --- Assemble ---------------------------------------------------------------
{
    echo "# Benchmark results"
    echo
    echo "_Généré par \`benchmark/run_bench.sh\` le $(date '+%Y-%m-%d %H:%M')._"
    echo
    printf '%s' "$frags"
} > "$OUT_MD"
echo "[bench] Résultats agrégés : $OUT_MD"
