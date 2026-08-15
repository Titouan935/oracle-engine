#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// platform.hpp — couche d'abstraction Windows/POSIX
//
// Fournit des wrappers cross-platform pour :
//   - Memory-mapped files (mmap)
//   - Allocation de memoire virtuelle (VirtualAlloc / mmap anonymous)
//   - Chargement de DLL/SO dynamiques (LoadLibrary / dlopen)
//   - Execution de processus (CreateProcess / fork+exec)
//   - Info RAM systeme
//
// Usage : inclure ce header au lieu de <windows.h> dans les modules portables.
// Chaque module utilise les fonctions platform::* au lieu des API natives.
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <cstdint>
#include <cstddef>

namespace platform {

// ═══════════════════════════════════════════════════════════════════════════
// Memory-mapped files
// ═══════════════════════════════════════════════════════════════════════════

struct MappedFile {
    const uint8_t* data = nullptr;
    size_t         size = 0;
    void*          handle1 = nullptr;  // hFile (Win) or fd cast (POSIX)
    void*          handle2 = nullptr;  // hMap  (Win) or unused
};

// Ouvre et mappe un fichier en lecture seule
// Retourne true si ok, remplit mf
bool mmap_open(const std::string& path, MappedFile& mf);

// Ferme le mapping
void mmap_close(MappedFile& mf);

// ═══════════════════════════════════════════════════════════════════════════
// Allocation memoire virtuelle (grandes reserves contigues)
// ═══════════════════════════════════════════════════════════════════════════

// Reserve + commit un bloc de memoire
// Retourne nullptr si echec
void* vmem_alloc(size_t bytes);

// Libere un bloc alloue par vmem_alloc
void vmem_free(void* ptr, size_t bytes);

// Prefetch — demande a l'OS de precharger les pages en RAM
void vmem_prefetch(const void* addr, size_t bytes);

// ═══════════════════════════════════════════════════════════════════════════
// Info RAM systeme
// ═══════════════════════════════════════════════════════════════════════════

struct MemInfo {
    uint64_t total_bytes     = 0;
    uint64_t available_bytes = 0;
};

MemInfo get_mem_info();

// ═══════════════════════════════════════════════════════════════════════════
// DLL / Shared Object dynamique
// ═══════════════════════════════════════════════════════════════════════════

using DllHandle = void*;
using FuncPtr   = void*;

DllHandle dll_load(const std::string& path);
FuncPtr   dll_sym(DllHandle h, const std::string& name);
void      dll_unload(DllHandle h);
std::string dll_error();  // Dernier message d'erreur

// ═══════════════════════════════════════════════════════════════════════════
// Execution de processus (capture stdout+stderr, timeout)
// ═══════════════════════════════════════════════════════════════════════════

struct ProcessResult {
    int         exit_code = -1;
    std::string output;
    bool        timed_out = false;
};

// Execute une commande, attend la fin, capture stdout+stderr
// timeout_ms = 0 → pas de timeout
ProcessResult run_process(const std::string& cmd, int timeout_ms = 0);

// Variante shell : encapsule la commande dans le shell natif
// (cmd.exe /C sur Windows, /bin/sh -c sur POSIX)
// Utiliser a la place de run_process quand la commande contient du shell (pipes, redirections, etc.)
ProcessResult run_shell(const std::string& cmd, int timeout_ms = 0);

// ═══════════════════════════════════════════════════════════════════════════
// Chemins portables
// ═══════════════════════════════════════════════════════════════════════════

// Combine un repertoire et un nom de fichier avec le separateur natif
std::string path_join(const std::string& dir, const std::string& file);

// ═══════════════════════════════════════════════════════════════════════════
// Compilation portable
// ═══════════════════════════════════════════════════════════════════════════

// Retourne les flags g++ pour compiler une shared library sur la plateforme
// Ex: "-shared -O2 -std=c++17" (Win) ou "-shared -fPIC -O2 -std=c++17" (POSIX)
std::string shared_compile_flags();

// ═══════════════════════════════════════════════════════════════════════════
// Restart applicatif
// ═══════════════════════════════════════════════════════════════════════════

// Ecrit et lance un script natif qui attend 2s puis relance exe_path
// (.bat sur Windows, .sh sur POSIX)
bool schedule_restart(const std::string& exe_path);

// ═══════════════════════════════════════════════════════════════════════════
// HTTP GET portable
// ═══════════════════════════════════════════════════════════════════════════

struct HttpResponse {
    std::string body;
    int         status_code = 0;
    bool        success     = false;
};

// HTTP GET — WinHTTP sur Windows, sockets POSIX (HTTP) + curl (HTTPS) sur POSIX
// max_bytes : taille maximale du body retourne (tronque si depasse)
HttpResponse http_get(const std::string& url, int timeout_ms = 8000,
                      size_t max_bytes = 512 * 1024);

// ═══════════════════════════════════════════════════════════════════════════
// Console UTF-8
// ═══════════════════════════════════════════════════════════════════════════

void set_console_utf8();

// Extension de fichier partagee (.dll / .dylib / .so)
#ifdef _WIN32
static constexpr const char* DLL_EXT = ".dll";
static constexpr const char* EXE_EXT = ".exe";
static constexpr char        PATH_SEP = '\\';
#elif defined(__APPLE__)
static constexpr const char* DLL_EXT = ".dylib";
static constexpr const char* EXE_EXT = "";
static constexpr char        PATH_SEP = '/';
#else
static constexpr const char* DLL_EXT = ".so";
static constexpr const char* EXE_EXT = "";
static constexpr char        PATH_SEP = '/';
#endif

} // namespace platform
