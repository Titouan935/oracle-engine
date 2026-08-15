#pragma once
// ══════════════════════════════════════════════════════════════════════════
//  MachineProfile — le core s'adapte à la machine
//
//  Analyse le PC (cœurs CPU, jeu d'instructions SIMD, RAM) et en déduit les
//  réglages pour exploiter le LLM au maximum sur N'IMPORTE QUELLE machine :
//    • nombre de threads d'inférence (OpenMP) adapté aux cœurs physiques
//    • stratégie mémoire (précharger le modèle en RAM vs mmap direct)
//
//  Zéro dépendance module — juste l'OS + std. Détecté une fois au chargement.
// ══════════════════════════════════════════════════════════════════════════
#include <string>
#include <cstdint>

struct MachineProfile {
    // CPU
    int  logical_cores  = 1;   // threads matériels (avec hyperthreading)
    int  physical_cores = 1;   // vrais cœurs
    // SIMD
    bool avx2   = false;
    bool avx512 = false;
    bool fma    = false;
    bool neon   = false;
    // Mémoire
    uint64_t ram_total_mb = 0;
    uint64_t ram_avail_mb = 0;
    // Réglage déduit
    int  recommended_threads = 1;   // pour l'inférence (mémoire-bound → cœurs physiques)

    // La RAM libre permet-elle de précharger un modèle de `model_bytes` ?
    // (préchargé = pas de page faults = beaucoup plus rapide)
    bool can_preload(uint64_t model_bytes) const;

    // Résumé lisible pour le log.
    std::string describe() const;
};

// Analyse la machine (CPU, SIMD, RAM).
MachineProfile detect_machine();

// Applique le profil : règle le nombre de threads OpenMP pour l'inférence.
// À appeler une fois avant/à l'inférence.
void apply_machine_profile(const MachineProfile& p);
