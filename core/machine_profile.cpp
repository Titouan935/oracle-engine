#include "core/machine_profile.hpp"
#include "core/platform.hpp"
#include "bug/logger.hpp"

#include <thread>
#include <cstdio>
#include <algorithm>

#ifdef _WIN32
  #include <windows.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif

// ── Cœurs physiques ────────────────────────────────────────────────────────
static int physical_core_count(int logical_fallback) {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len == 0) return logical_fallback;
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(
        len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buf.data(), &len)) return logical_fallback;
    int count = 0;
    for (auto& info : buf)
        if (info.Relationship == RelationProcessorCore) count++;
    return count > 0 ? count : logical_fallback;
#else
    // POSIX : heuristique (hyperthreading fréquent = logical/2), bornée.
    return std::max(1, logical_fallback / 2);
#endif
}

// ── Détection SIMD ─────────────────────────────────────────────────────────
static void detect_simd(MachineProfile& p) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #if defined(__GNUC__)
    __builtin_cpu_init();
    p.avx2   = __builtin_cpu_supports("avx2");
    p.fma    = __builtin_cpu_supports("fma");
    p.avx512 = __builtin_cpu_supports("avx512f");
  #endif
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
    p.neon = true;
#endif
}

MachineProfile detect_machine() {
    MachineProfile p;

    unsigned hc = std::thread::hardware_concurrency();
    p.logical_cores  = hc > 0 ? (int)hc : 1;
    p.physical_cores = physical_core_count(p.logical_cores);

    detect_simd(p);

    auto mi = platform::get_mem_info();
    p.ram_total_mb = mi.total_bytes     / (1024 * 1024);
    p.ram_avail_mb = mi.available_bytes / (1024 * 1024);

    // Inférence LLM = surtout memory-bound (GEMV quantisé). Les cœurs physiques
    // saturent déjà la bande passante ; les hyperthreads ajoutent de la
    // contention. On recommande donc les cœurs physiques.
    p.recommended_threads = std::max(1, p.physical_cores);

    return p;
}

bool MachineProfile::can_preload(uint64_t model_bytes) const {
    // Besoin de ~120 % du modèle en RAM libre pour précharger sans thrashing.
    const uint64_t need_mb = (model_bytes / (1024 * 1024)) * 6 / 5;
    return ram_avail_mb >= need_mb;
}

std::string MachineProfile::describe() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CPU %d cœurs phys / %d threads | SIMD:%s%s%s%s | RAM %llu/%llu MB | threads inf=%d",
        physical_cores, logical_cores,
        avx2 ? " AVX2" : "", fma ? " FMA" : "", avx512 ? " AVX512" : "",
        neon ? " NEON" : (avx2 ? "" : " (scalaire)"),
        (unsigned long long)ram_avail_mb, (unsigned long long)ram_total_mb,
        recommended_threads);
    return buf;
}

void apply_machine_profile(const MachineProfile& p) {
#ifdef _OPENMP
    omp_set_num_threads(p.recommended_threads);
    LOG_INFO("Machine", "OpenMP réglé sur " + std::to_string(p.recommended_threads) +
             " threads (cœurs physiques)");
#endif
    LOG_INFO("Machine", p.describe());
}
