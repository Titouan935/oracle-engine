# Benchmarks

> Les chiffres ci-dessous se génèrent avec le harnais reproductible fourni.
> Ne remplacer les `[…]` qu'avec une sortie réelle de `oracle_bench` — un
> nombre inventé se fait démonter en commentaire dans l'heure.

## Setup

Deux machines. Les chiffres publiés plus bas viennent de la **Machine A**
(machine de développement, bas de gamme). La **Machine B** (Mac Pro, cible de
déploiement) reste à mesurer et donnera les chiffres du modèle principal 7B.

### Machine A — dev box (chiffres actuels)

| | |
|---|---|
| Machine | Intel **Pentium Gold 8505** — 5 cœurs (1 P + 4 E) / 6 threads logiques, 7.6 GB RAM |
| OS | Windows 11 Pro (build 26200) |
| Compiler | GNU (g++) 15.2.0 |
| Flags | `-O3 -DNDEBUG -mavx2 -mfma -march=native` |
| Model | `llama-3.2-3b-q4k.gguf` (draft model, Q4_K, 1.9 GB) |
| Baseline | llama.cpp — **pas encore mesuré** sur cette machine |

> ⚠️ CPU d'entrée de gamme (1 seul cœur performance). Ces chiffres ne sont **pas
> représentatifs** de la cible de déploiement. Le run a demandé 8 threads sur
> 6 threads logiques (légère sur-souscription).

### Machine B — Mac Pro (cible, à mesurer)

| | |
|---|---|
| Machine | Mac Pro — [CPU, cœurs, RAM] |
| OS | macOS [version] |
| Compiler | rempli automatiquement (`-O3 -DNDEBUG -mcpu=apple-m1`) |
| Model | `qwen2.5-7b-instruct-q4km.gguf` (modèle principal) |
| Baseline | llama.cpp [commit hash], built with [flags] |

## Method

Le harnais `oracle_bench` mesure séparément et de façon déterministe :

- **Prefill** — débit du traitement batché du prompt (`forward_prefill`, GEMM weight-stationary).
- **Génération** — débit de la décode autorégressive, **greedy (argmax)** pour une
  sortie déterministe indépendante du sampler.
- **Peak RSS** — mémoire résidente maximale du process.

Protocole :

- Prompt fixe, tuilé jusqu'à `-p` tokens ; génération de `-g` tokens.
- `-r` runs, **médiane** reportée, 1er run jeté (warmup) en plus de `model.warmup()`.
- KV cache remis à zéro (`reset`) avant chaque run.
- Nombre de threads fixé (`-t`), imprimé dans le rapport.
- **Checksum FNV-1a** des tokens générés imprimé : deux runs identiques → même hash
  (preuve de déterminisme).

## Reproduce

Prérequis : assez de RAM **libre** pour charger le modèle sans paging
(≈ 2 GB pour le 3B Q4, ≈ 5 GB pour le 7B Q4_K_M). Sinon le moteur bascule en
mmap direct et les chiffres s'effondrent.

**macOS / Linux :**

```bash
./benchmark/run_bench.sh
# ou, ciblé :
PROMPT=128 GEN=128 RUNS=5 THREADS=8 \
  MODELS="models/qwen2.5-7b-instruct-q4km.gguf" ./benchmark/run_bench.sh
```

**Windows (MinGW64) :**

```powershell
.\benchmark\run_bench.ps1 -Prompt 128 -Gen 128 -Runs 5 -Threads 8
```

**Un seul modèle, à la main :**

```bash
./build/oracle_bench -m models/qwen2.5-7b-instruct-q4km.gguf \
    -p 128 -g 128 -r 5 -t 8 --md frag.md --json result.json
```

Les résultats agrégés sont écrits dans `benchmark/results.md` (fragment Markdown
prêt à coller ci-dessous) et un JSON par modèle.

## Results

### Machine A — Intel Pentium Gold 8505, `llama-3.2-3b-q4k`

Poids préchargés en RAM (aucun page fault pendant la mesure). Médiane sur 4 runs
mesurés (1er jeté en warmup). Prompt = 127 tokens, génération = 128 tokens, greedy,
8 threads. Checksum déterministe des tokens générés : `47f5b9ec088824ec`.

| Engine | Model | Prefill (tok/s) | Generation (tok/s) | Peak RSS | Threads |
|---|---|---|---|---|---|
| ORACLE | llama-3.2-3b-q4k | **1.8** | **1.0** | 3386 MB | 8 |
| llama.cpp | llama-3.2-3b-q4k | _à mesurer (`compare_llama.sh`)_ | _à mesurer_ | — | 8 |

Variance des runs mesurés : prefill 1.8–1.9 tok/s, génération 1.0 tok/s (constante) —
la faible variance confirme l'absence de paging.

### Machine B — Mac Pro, `qwen2.5-7b-instruct-q4km`

_À mesurer._ Cette machine A (8 GB RAM) ne peut pas préallouer le 7B (4.5 GB) ;
les chiffres du modèle principal + la baseline llama.cpp viendront du Mac Pro.

## Honest notes

- **Ces chiffres sont sur un CPU bas de gamme (Pentium Gold 8505, 1 seul cœur
  performance)**, pas sur la cible de déploiement. Ils mesurent le moteur, pas le
  matériel final.
- **ORACLE est nettement plus lent que llama.cpp.** Le ratio exact n'est pas encore
  mesuré sur cette machine (lancer `compare_llama.sh`), mais l'ordre de grandeur
  attendu est de plusieurs fois plus lent : llama.cpp a des kernels SIMD écrits à la
  main et des années d'optimisation, ORACLE est un moteur from-scratch sans
  dépendance. C'est défendable — mais le cacher serait le seul vrai risque
  réputationnel.
- **Le prefill n'est que ~1.8× plus rapide que la génération** alors que le batching
  `gemm_q` devrait donner bien plus : signe qu'il reste de la marge d'optimisation
  côté matmul quantisé (là où se cache l'essentiel du temps mur).

## Notes de mesure

- La comparaison à llama.cpp doit se faire sur la **même machine**, même modèle,
  même quantization, même thread count, prompt identique.
- La génération d'ORACLE est mesurée en **greedy** ; pour comparer, lancer
  llama.cpp avec `--temp 0` (ou `-n` fixe et sampling neutralisé).
- Le prefill d'ORACLE utilise `forward_prefill` (batché). Comparer au
  « prompt eval » de llama.cpp (`llama-bench -p <N> -n <M>`).
