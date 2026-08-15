#include "core/platform.hpp"
#include <cstring>
#include <fstream>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════
// Implementations WINDOWS
// ═══════════════════════════════════════════════════════════════════════════

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

namespace platform {

// ── mmap ────────────────────────────────────────────────────────────────
bool mmap_open(const std::string& path, MappedFile& mf) {
    std::string native_path = path;
    for (char& c : native_path) if (c == '/') c = '\\';

    HANDLE hf = CreateFileA(native_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) return false;

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz)) { CloseHandle(hf); return false; }

    HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) { CloseHandle(hf); return false; }

    void* view = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!view) { CloseHandle(hm); CloseHandle(hf); return false; }

    mf.data    = (const uint8_t*)view;
    mf.size    = (size_t)sz.QuadPart;
    mf.handle1 = (void*)hf;
    mf.handle2 = (void*)hm;
    return true;
}

void mmap_close(MappedFile& mf) {
    if (mf.data)    UnmapViewOfFile(mf.data);
    if (mf.handle2) CloseHandle((HANDLE)mf.handle2);
    if (mf.handle1) CloseHandle((HANDLE)mf.handle1);
    mf = {};
}

// ── vmem ────────────────────────────────────────────────────────────────
void* vmem_alloc(size_t bytes) {
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void vmem_free(void* ptr, size_t /*bytes*/) {
    if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);
}

void vmem_prefetch(const void* addr, size_t bytes) {
    using PFN_PVM = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static auto pfn = (PFN_PVM)GetProcAddress(
        GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory");
    if (pfn) {
        WIN32_MEMORY_RANGE_ENTRY e;
        e.VirtualAddress = (PVOID)addr;
        e.NumberOfBytes  = (SIZE_T)bytes;
        pfn(GetCurrentProcess(), 1, &e, 0);
    }
}

// ── mem info ────────────────────────────────────────────────────────────
MemInfo get_mem_info() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);
    return { ms.ullTotalPhys, ms.ullAvailPhys };
}

// ── dll ─────────────────────────────────────────────────────────────────
DllHandle dll_load(const std::string& path) {
    return (DllHandle)LoadLibraryA(path.c_str());
}

FuncPtr dll_sym(DllHandle h, const std::string& name) {
    return (FuncPtr)GetProcAddress((HMODULE)h, name.c_str());
}

void dll_unload(DllHandle h) {
    if (h) FreeLibrary((HMODULE)h);
}

std::string dll_error() {
    DWORD err = GetLastError();
    return "Win32 error " + std::to_string(err);
}

// ── process ─────────────────────────────────────────────────────────────
ProcessResult run_process(const std::string& cmd, int timeout_ms) {
    ProcessResult res;

    SECURITY_ATTRIBUTES sa;
    sa.nLength              = sizeof(sa);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE rd, wr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) {
        res.output = "CreatePipe failed";
        return res;
    }
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = {};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError  = wr;

    PROCESS_INFORMATION pi = {};
    BOOL ok = CreateProcessA(nullptr, const_cast<char*>(cmd.c_str()),
                             nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);

    if (!ok) {
        CloseHandle(rd);
        res.output = "CreateProcess failed: " + std::to_string(GetLastError());
        return res;
    }

    // Lire stdout/stderr
    char buf[4096];
    DWORD n;
    while (ReadFile(rd, buf, sizeof(buf) - 1, &n, nullptr) && n > 0) {
        buf[n] = '\0';
        res.output += buf;
    }
    CloseHandle(rd);

    DWORD wait_ms = (timeout_ms > 0) ? (DWORD)timeout_ms : INFINITE;
    DWORD wr2 = WaitForSingleObject(pi.hProcess, wait_ms);
    if (wr2 == WAIT_TIMEOUT) {
        TerminateProcess(pi.hProcess, 1);
        res.timed_out = true;
    }

    DWORD ec = 0;
    GetExitCodeProcess(pi.hProcess, &ec);
    res.exit_code = (int)ec;

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return res;
}

// ── run_shell ───────────────────────────────────────────────────────────
ProcessResult run_shell(const std::string& cmd, int timeout_ms) {
    return run_process("cmd.exe /C " + cmd, timeout_ms);
}

// ── path_join ───────────────────────────────────────────────────────────
std::string path_join(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '\\' || last == '/') return dir + file;
    return dir + '\\' + file;
}

// ── shared_compile_flags ────────────────────────────────────────────────
std::string shared_compile_flags() {
    return "-shared -O2 -std=c++17";
}

// ── schedule_restart ────────────────────────────────────────────────────
bool schedule_restart(const std::string& exe_path) {
    std::string script_path = "oracle_restart.bat";
    {
        std::ofstream f(script_path);
        if (!f.is_open()) return false;
        f << "@echo off\n"
          << "timeout /t 2 /nobreak >nul\n"
          << "start \"\" \"" << exe_path << "\"\n"
          << "del \"%~f0\"\n";
    }
    auto res = run_process("cmd.exe /C oracle_restart.bat", 0);
    return (res.exit_code >= 0);
}

// ── http_get (WinHTTP) ─────────────────────────────────────────────────
HttpResponse http_get(const std::string& url, int timeout_ms, size_t max_bytes) {
    HttpResponse res;

    std::wstring wurl(url.begin(), url.end());

    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof(uc);
    wchar_t host[256]{}, path[4096]{};
    uc.lpszHostName     = host;
    uc.dwHostNameLength = 256;
    uc.lpszUrlPath      = path;
    uc.dwUrlPathLength  = 4096;

    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) return res;

    bool is_https = (uc.nScheme == INTERNET_SCHEME_HTTPS);

    HINTERNET session = WinHttpOpen(L"ORACLE/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                     WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) return res;
    WinHttpSetTimeouts(session, timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    HINTERNET connect = WinHttpConnect(session, host, uc.nPort, 0);
    if (!connect) { WinHttpCloseHandle(session); return res; }

    DWORD flags = is_https ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET request = WinHttpOpenRequest(connect, L"GET", path, nullptr,
                                            WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!request) {
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        return res;
    }

    // Ignorer les erreurs SSL pour SearXNG local
    if (is_https) {
        DWORD sec_flags = SECURITY_FLAG_IGNORE_UNKNOWN_CA
                        | SECURITY_FLAG_IGNORE_CERT_DATE_INVALID
                        | SECURITY_FLAG_IGNORE_CERT_CN_INVALID
                        | SECURITY_FLAG_IGNORE_CERT_WRONG_USAGE;
        WinHttpSetOption(request, WINHTTP_OPTION_SECURITY_FLAGS,
                         &sec_flags, sizeof(sec_flags));
    }

    if (!WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                             WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request, nullptr)) {
        WinHttpCloseHandle(request);
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        return res;
    }

    // Status code
    DWORD sc = 0, sc_size = sizeof(sc);
    WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                         WINHTTP_HEADER_NAME_BY_INDEX, &sc, &sc_size,
                         WINHTTP_NO_HEADER_INDEX);
    res.status_code = static_cast<int>(sc);

    // Body
    char buf[8192];
    DWORD rd = 0;
    while (WinHttpReadData(request, buf, sizeof(buf), &rd) && rd > 0) {
        res.body.append(buf, rd);
        if (res.body.size() > max_bytes) break;
        rd = 0;
    }

    WinHttpCloseHandle(request);
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);

    if (res.body.size() > max_bytes) res.body.resize(max_bytes);
    res.success = (res.status_code >= 200 && res.status_code < 400);
    return res;
}

// ── console ─────────────────────────────────────────────────────────────
void set_console_utf8() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(hOut, &mode);
    SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}

} // namespace platform

// ═══════════════════════════════════════════════════════════════════════════
// Implementations POSIX (Linux / macOS)
// ═══════════════════════════════════════════════════════════════════════════

#else

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

namespace platform {

// ── mmap ────────────────────────────────────────────────────────────────
bool mmap_open(const std::string& path, MappedFile& mf) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return false; }

    void* ptr = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); return false; }

    mf.data    = (const uint8_t*)ptr;
    mf.size    = (size_t)st.st_size;
    mf.handle1 = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
    mf.handle2 = nullptr;
    return true;
}

void mmap_close(MappedFile& mf) {
    if (mf.data) munmap(const_cast<uint8_t*>(mf.data), mf.size);
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(mf.handle1));
    if (fd >= 0) close(fd);
    mf = {};
}

// ── vmem ────────────────────────────────────────────────────────────────
void* vmem_alloc(size_t bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? nullptr : p;
}

void vmem_free(void* ptr, size_t bytes) {
    if (ptr) munmap(ptr, bytes);
}

void vmem_prefetch(const void* addr, size_t bytes) {
    madvise(const_cast<void*>(addr), bytes, MADV_WILLNEED);
}

// ── mem info ────────────────────────────────────────────────────────────
MemInfo get_mem_info() {
    MemInfo info;
#ifdef __APPLE__
    // macOS: sysctl + Mach VM stats
    int64_t phys = 0;
    size_t sz = sizeof(phys);
    sysctlbyname("hw.memsize", &phys, &sz, nullptr, 0);
    info.total_bytes = (uint64_t)phys;

    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          (host_info64_t)&vm, &count) == KERN_SUCCESS) {
        vm_size_t page_size = 0;
        host_page_size(host, &page_size);
        info.available_bytes = (uint64_t)(vm.free_count + vm.inactive_count) * page_size;
    }
#else
    // Linux: /proc/meminfo
    std::ifstream f("/proc/meminfo");
    if (!f.is_open()) return info;
    std::string line;
    while (std::getline(f, line)) {
        uint64_t val = 0;
        if (line.find("MemTotal:") == 0) {
            sscanf(line.c_str(), "MemTotal: %lu kB", &val);
            info.total_bytes = val * 1024;
        } else if (line.find("MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %lu kB", &val);
            info.available_bytes = val * 1024;
        }
    }
#endif
    return info;
}

// ── dll (shared objects) ────────────────────────────────────────────────
DllHandle dll_load(const std::string& path) {
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}

FuncPtr dll_sym(DllHandle h, const std::string& name) {
    return dlsym(h, name.c_str());
}

void dll_unload(DllHandle h) {
    if (h) dlclose(h);
}

std::string dll_error() {
    const char* e = dlerror();
    return e ? std::string(e) : "";
}

// ── process ─────────────────────────────────────────────────────────────
ProcessResult run_process(const std::string& cmd, int timeout_ms) {
    ProcessResult res;

    // popen pour capturer stdout+stderr
    std::string full_cmd = cmd + " 2>&1";
    FILE* fp = popen(full_cmd.c_str(), "r");
    if (!fp) {
        res.output = "popen failed: " + std::string(strerror(errno));
        return res;
    }

    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) {
        res.output += buf;
    }

    int status = pclose(fp);
    if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    } else {
        res.exit_code = -1;
    }

    return res;
}

// ── run_shell ───────────────────────────────────────────────────────────
ProcessResult run_shell(const std::string& cmd, int timeout_ms) {
    return run_process("/bin/sh -c '" + cmd + "'", timeout_ms);
}

// ── path_join ───────────────────────────────────────────────────────────
std::string path_join(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/') return dir + file;
    return dir + '/' + file;
}

// ── shared_compile_flags ────────────────────────────────────────────────
std::string shared_compile_flags() {
    return "-shared -fPIC -O2 -std=c++17";
}

// ── schedule_restart ────────────────────────────────────────────────────
bool schedule_restart(const std::string& exe_path) {
    std::string script_path = "/tmp/oracle_restart.sh";
    {
        std::ofstream f(script_path);
        if (!f.is_open()) return false;
        f << "#!/bin/sh\n"
          << "sleep 2\n"
          << "nohup \"" << exe_path << "\" &\n"
          << "rm -f \"$0\"\n";
    }
    run_process("chmod +x " + script_path, 5000);
    run_process(script_path + " &", 0);
    return true;
}

// ── http_get (sockets POSIX + curl fallback) ───────────────────────────

// Parse URL en composants
struct ParsedUrl_ {
    std::string scheme;  // "http" ou "https"
    std::string host;
    int         port = 80;
    std::string path;    // "/search?q=..."
};

static ParsedUrl_ parse_url_(const std::string& url) {
    ParsedUrl_ u;
    size_t pos = 0;

    auto scheme_end = url.find("://");
    if (scheme_end != std::string::npos) {
        u.scheme = url.substr(0, scheme_end);
        std::transform(u.scheme.begin(), u.scheme.end(), u.scheme.begin(), ::tolower);
        pos = scheme_end + 3;
    } else {
        u.scheme = "http";
    }

    auto path_start = url.find('/', pos);
    std::string host_port;
    if (path_start != std::string::npos) {
        host_port = url.substr(pos, path_start - pos);
        u.path = url.substr(path_start);
    } else {
        host_port = url.substr(pos);
        u.path = "/";
    }

    auto colon = host_port.find(':');
    if (colon != std::string::npos) {
        u.host = host_port.substr(0, colon);
        try { u.port = std::stoi(host_port.substr(colon + 1)); } catch (...) {}
    } else {
        u.host = host_port;
        u.port = (u.scheme == "https") ? 443 : 80;
    }
    return u;
}

// HTTP GET via socket brut (HTTP uniquement, pas TLS)
static HttpResponse http_get_socket_(const ParsedUrl_& url, int timeout_ms,
                                      size_t max_bytes) {
    HttpResponse res;

    // Resolution DNS
    struct addrinfo hints{}, *ai = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(url.port);
    if (getaddrinfo(url.host.c_str(), port_str.c_str(), &hints, &ai) != 0)
        return res;

    int sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sock < 0) { freeaddrinfo(ai); return res; }

    // Timeout lecture/ecriture
    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    if (connect(sock, ai->ai_addr, ai->ai_addrlen) < 0) {
        close(sock); freeaddrinfo(ai); return res;
    }
    freeaddrinfo(ai);

    // Requete HTTP/1.0 (pas de chunked, connection close implicite)
    std::string req = "GET " + url.path + " HTTP/1.0\r\n"
                      "Host: " + url.host + "\r\n"
                      "User-Agent: ORACLE/1.0\r\n"
                      "Accept: */*\r\n"
                      "\r\n";

    if (send(sock, req.c_str(), req.size(), 0) < 0) {
        close(sock); return res;
    }

    // Reception jusqu'a fermeture connexion
    std::string raw;
    raw.reserve(32768);
    char buf[8192];
    ssize_t n;
    while ((n = recv(sock, buf, sizeof(buf), 0)) > 0) {
        raw.append(buf, static_cast<size_t>(n));
        if (raw.size() > max_bytes + 16384) break;  // marge pour headers
    }
    close(sock);

    if (raw.empty()) return res;

    // Parse status line : "HTTP/1.0 200 OK\r\n"
    auto sp = raw.find(' ');
    if (sp != std::string::npos) {
        try { res.status_code = std::stoi(raw.substr(sp + 1)); } catch (...) {}
    }

    // Separateur headers / body
    auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) return res;

    res.body = raw.substr(hdr_end + 4);
    if (res.body.size() > max_bytes) res.body.resize(max_bytes);
    res.success = (res.status_code >= 200 && res.status_code < 400);
    return res;
}

// HTTP(S) GET via curl subprocess (fallback HTTPS et erreurs socket)
static HttpResponse http_get_curl_(const std::string& url, int timeout_ms,
                                    size_t max_bytes) {
    HttpResponse res;

    int timeout_sec = std::max(1, timeout_ms / 1000);

    // curl -sL : silent + follow redirects
    // -w '\n%{http_code}' : status code en derniere ligne
    // --max-filesize : limite taille body
    std::string cmd = "curl -sL"
                      " --max-time " + std::to_string(timeout_sec) +
                      " --max-filesize " + std::to_string(max_bytes) +
                      " -w '\\n%{http_code}'"
                      " -A 'ORACLE/1.0'"
                      " '" + url + "' 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return res;

    std::string result;
    result.reserve(32768);
    char buf[8192];
    while (fgets(buf, sizeof(buf), pipe))
        result += buf;
    pclose(pipe);

    if (result.empty()) return res;

    // Extraire le status code de la derniere ligne
    auto nl = result.rfind('\n', result.size() > 1 ? result.size() - 2 : 0);
    if (nl != std::string::npos) {
        std::string code_str = result.substr(nl + 1);
        while (!code_str.empty() &&
               (code_str.back() == '\n' || code_str.back() == '\r'))
            code_str.pop_back();
        try { res.status_code = std::stoi(code_str); } catch (...) {}
        result.resize(nl);
    }

    if (result.size() > max_bytes) result.resize(max_bytes);
    res.body    = std::move(result);
    res.success = (res.status_code >= 200 && res.status_code < 400);
    return res;
}

HttpResponse http_get(const std::string& url, int timeout_ms, size_t max_bytes) {
    auto parsed = parse_url_(url);

    // HTTPS → curl (TLS necessite une lib)
    if (parsed.scheme == "https") {
        return http_get_curl_(url, timeout_ms, max_bytes);
    }

    // HTTP → socket brut, avec fallback curl si echec
    auto res = http_get_socket_(parsed, timeout_ms, max_bytes);
    if (!res.success && res.status_code == 0) {
        return http_get_curl_(url, timeout_ms, max_bytes);
    }
    return res;
}

// ── console ─────────────────────────────────────────────────────────────
void set_console_utf8() {
    // POSIX : rien a faire, UTF-8 est le defaut
}

} // namespace platform

#endif
