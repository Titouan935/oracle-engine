#include "gguf_parser.hpp"
#include "core/platform.hpp"
#include "core/ops.hpp"
#include "bug/logger.hpp"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// ══════════════════════════════════════════════════════════════════════════
//  Helpers lecture little-endian (Windows est LE — cast direct)
// ══════════════════════════════════════════════════════════════════════════
template<typename T>
static T rd(const uint8_t*& p) {
    T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
}

static uint64_t align32(uint64_t x) { return (x + 31) & ~uint64_t(31); }

#ifdef __ARM_NEON
static inline float hsum_neon(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}
#endif

#ifdef __AVX2__
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

// ══════════════════════════════════════════════════════════════════════════
//  TensorInfo
// ══════════════════════════════════════════════════════════════════════════
uint64_t TensorInfo::n_elems() const {
    if (shape.empty()) return 0;
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

uint64_t TensorInfo::n_bytes() const {
    uint64_t n = n_elems();
    switch (type) {
        case GGMLType::F32:  return n * 4;
        case GGMLType::F16:
        case GGMLType::BF16: return n * 2;
        case GGMLType::Q4_0: return (n / 32) * 18;        // 18 bytes / block 32
        case GGMLType::Q8_0: return (n / 32) * 34;        // 34 bytes / block 32
        case GGMLType::Q4_K: return (n / 256) * 144;      // 144 bytes / block 256
        case GGMLType::Q5_K: return (n / 256) * 176;
        case GGMLType::Q6_K: return (n / 256) * 210;      // 210 bytes / block 256
        case GGMLType::Q8_K: return (n / 256) * 292;
        default:             return 0;
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  Chargement / fermeture
// ══════════════════════════════════════════════════════════════════════════
bool GGUFParser::load(const std::string& path) {
    close();

    // Memory-map le fichier (cross-platform via platform.hpp)
    if (!platform::mmap_open(path, mmap_)) {
        LOG_ERR("GGUFParser", "Impossible d'ouvrir : " + path);
        return false;
    }
    data_  = mmap_.data;
    fsize_ = mmap_.size;

    if (!parse()) { close(); return false; }

    // Vérifie que TOUS les tenseurs ont un type déquantisable. Échec explicite
    // au chargement plutôt que des zéros silencieux au moment de l'inférence
    // (ex : un modèle contenant du Q5_K, non implémenté ici).
    for (const auto& [tname, ti] : tensors_) {
        switch (ti.type) {
            case GGMLType::F32:  case GGMLType::F16:  case GGMLType::BF16:
            case GGMLType::Q4_0: case GGMLType::Q8_0:
            case GGMLType::Q4_K: case GGMLType::Q6_K:
                break;  // type supporté
            default:
                LOG_ERR("GGUFParser",
                        "Type quantisé non supporté pour le tenseur '" + tname
                        + "' : " + std::to_string((uint32_t)ti.type)
                        + " (déquantisation non implémentée)");
                close();
                return false;
        }
    }

    // Précharge la section données (poids) en RAM pour éviter les page faults mmap.
    const size_t data_bytes = fsize_ - data_start_;
    auto mi = platform::get_mem_info();
    LOG_INFO("GGUFParser",
        "RAM: " + std::to_string(mi.available_bytes / 1024 / 1024) + " MB libres, "
        "données: " + std::to_string(data_bytes / 1024 / 1024) + " MB");

    // Strategie : allouer un buffer contigu et copier les poids dedans
    // pour eviter les page faults aleatoires du mmap pendant l'inference.
    // Skip si RAM insuffisante (besoin de 120% de data_bytes pour eviter thrashing).
    void* vbuf = nullptr;
    if (mi.available_bytes > data_bytes + data_bytes / 5) {
        vbuf = platform::vmem_alloc(data_bytes);
    }
    if (vbuf) {
        std::memcpy(vbuf, data_ + data_start_, data_bytes);
        data_virt_ = vbuf;
        data_virt_size_ = data_bytes;
        data_base_ = (const uint8_t*)vbuf;
        LOG_INFO("GGUFParser", "Poids préchargés en RAM — "
                 + std::to_string(data_bytes / 1024 / 1024) + " MB");
    } else {
        // Fallback : acceder directement via mmap (page faults a la demande)
        data_base_ = data_ + data_start_;
        LOG_WARN("GGUFParser", "RAM insuffisante — mode mmap direct ("
                 + std::to_string(data_bytes / 1024 / 1024) + " MB, page faults a la demande)");
    }

    LOG_INFO("GGUFParser", path + " chargé — "
             + std::to_string(n_tensors_) + " tenseurs, "
             + std::to_string(meta_.size()) + " métadonnées");
    return true;
}

void GGUFParser::close() {
    if (data_virt_) {
        platform::vmem_free(data_virt_, data_virt_size_);
        data_virt_ = nullptr;
        data_virt_size_ = 0;
    }
    data_base_ = nullptr;
    data_ = nullptr;
    platform::mmap_close(mmap_);
    meta_.clear();
    tensors_.clear();
    version_ = n_tensors_ = data_start_ = 0;
    fsize_ = 0;
}

// ══════════════════════════════════════════════════════════════════════════
//  Parsing principal
// ══════════════════════════════════════════════════════════════════════════
bool GGUFParser::parse() {
    const uint8_t* end = data_ + fsize_;
    const uint8_t* p   = data_;

    // Magie
    if (fsize_ < 4 || std::memcmp(p, "GGUF", 4) != 0) {
        LOG_ERR("GGUFParser", "Magic GGUF absent");
        return false;
    }
    p += 4;

    version_   = rd<uint32_t>(p);
    n_tensors_ = rd<uint64_t>(p);
    uint64_t n_kv = rd<uint64_t>(p);

    if (version_ < 1 || version_ > 3) {
        LOG_WARN("GGUFParser", "Version GGUF inattendue : " + std::to_string(version_));
    }

    // Métadonnées
    for (uint64_t i = 0; i < n_kv; ++i) {
        p = read_kv(p, end);
        if (!p) return false;
    }

    // Infos tenseurs
    for (uint64_t i = 0; i < n_tensors_; ++i) {
        p = read_tensor_info(p, end);
        if (!p) return false;
    }

    // Début des données (aligné 32 octets depuis le début du fichier)
    data_start_ = align32((uint64_t)(p - data_));
    return true;
}

// ── Lecture d'une paire clé-valeur ────────────────────────────────────────
const uint8_t* GGUFParser::read_kv(const uint8_t* p, const uint8_t* end) {
    std::string key = read_str(p, end);
    if (p + 4 > end) return nullptr;   // besoin de 4 octets pour le `kind`

    uint32_t kind = rd<uint32_t>(p);
    MetaValue v = read_value(p, end, kind);

    // Fixe le kind : pour les scalaires, le surpasse d'après `kind` GGUF ;
    // pour les tableaux (kind=9), read_value l'a déjà positionné correctement.
    switch (kind) {
        case 0:  v.kind = MetaValue::U8;      break;
        case 1:  v.kind = MetaValue::I8;      break;
        case 2:  v.kind = MetaValue::U16;     break;
        case 3:  v.kind = MetaValue::I16;     break;
        case 4:  v.kind = MetaValue::U32;     break;
        case 5:  v.kind = MetaValue::I32;     break;
        case 6:  v.kind = MetaValue::F32;     break;
        case 7:  v.kind = MetaValue::BOOL;    break;
        case 8:  v.kind = MetaValue::STR;     break;
        case 9:  break;   // Array : kind déjà positionné par read_value (ARR_STR, ARR_I32, ...)
        case 10: v.kind = MetaValue::U64;     break;
        case 11: v.kind = MetaValue::I64;     break;
        case 12: v.kind = MetaValue::F64;     break;
        default: v.kind = MetaValue::SKIP;    break;
    }

    meta_.emplace(std::move(key), std::move(v));
    return p;
}

// ── Lecture d'une valeur typée ────────────────────────────────────────────
MetaValue GGUFParser::read_value(const uint8_t*& p, const uint8_t* end, uint32_t kind) {
    MetaValue v;
    // Garde-fou : vérifie qu'il reste au moins `n` octets avant de lire.
    // En cas de dépassement (fichier tronqué/corrompu), on positionne p = end
    // pour faire échouer proprement le reste du parsing au lieu de lire hors limites.
    auto avail = [&](size_t n) -> bool {
        if ((size_t)(end - p) < n) { p = end; v.kind = MetaValue::SKIP; return false; }
        return true;
    };
    switch (kind) {
        case 0:  if (avail(1)) v.u8  = rd<uint8_t>(p);  break;
        case 1:  if (avail(1)) v.i8  = rd<int8_t>(p);   break;
        case 2:  if (avail(2)) v.u16 = rd<uint16_t>(p); break;
        case 3:  if (avail(2)) v.i16 = rd<int16_t>(p);  break;
        case 4:  if (avail(4)) v.u32 = rd<uint32_t>(p); break;
        case 5:  if (avail(4)) v.i32 = rd<int32_t>(p);  break;
        case 6:  if (avail(4)) v.f32 = rd<float>(p);    break;
        case 7:  if (avail(1)) v.b   = (bool)rd<uint8_t>(p); break;
        case 8:  v.str = read_str(p, end);     break;
        case 10: if (avail(8)) v.u64 = rd<uint64_t>(p); break;
        case 11: if (avail(8)) v.i64 = rd<int64_t>(p);  break;
        case 12: if (avail(8)) v.f64 = rd<double>(p);   break;
        case 9: {  // Array
            if (!avail(12)) break;   // 4 (type) + 8 (len)
            uint32_t arr_type = rd<uint32_t>(p);
            uint64_t arr_len  = rd<uint64_t>(p);

            if (arr_type == 8) {
                v.kind = MetaValue::ARR_STR;
                for (uint64_t i = 0; i < arr_len && p < end; ++i)
                    v.arr_str.push_back(read_str(p, end));
            } else if (arr_type == 5) {
                v.kind = MetaValue::ARR_I32;
                for (uint64_t i = 0; i < arr_len && p + 4 <= end; ++i)
                    v.arr_i32.push_back(rd<int32_t>(p));
            } else if (arr_type == 4) {
                v.kind = MetaValue::ARR_U32;
                for (uint64_t i = 0; i < arr_len && p + 4 <= end; ++i)
                    v.arr_u32.push_back(rd<uint32_t>(p));
            } else if (arr_type == 6) {
                v.kind = MetaValue::ARR_F32;
                for (uint64_t i = 0; i < arr_len && p + 4 <= end; ++i)
                    v.arr_f32.push_back(rd<float>(p));
            } else {
                // Tableau d'un type non géré : skip
                for (uint64_t i = 0; i < arr_len && p < end; ++i) {
                    MetaValue tmp;
                    tmp = read_value(p, end, arr_type);
                }
                v.kind = MetaValue::SKIP;
            }
            break;
        }
        default:
            // Type inconnu — on ne peut pas avancer sans taille connue
            LOG_WARN("GGUFParser", "Type KV inconnu : " + std::to_string(kind));
            v.kind = MetaValue::SKIP;
            break;
    }
    return v;
}

// ── Lecture d'une chaîne GGUF (uint64 len + octets) ──────────────────────
std::string GGUFParser::read_str(const uint8_t*& p, const uint8_t* end) {
    if (p + 8 > end) return {};
    uint64_t len = rd<uint64_t>(p);
    // Borne par les octets restants du fichier (plus de plafond arbitraire :
    // certains chat_template dépassent 64 Ko). Un `len` incohérent = corrompu.
    if (len > (uint64_t)(end - p)) return {};
    std::string s((const char*)p, len);
    p += len;
    return s;
}

// ── Lecture des infos d'un tenseur ───────────────────────────────────────
const uint8_t* GGUFParser::read_tensor_info(const uint8_t* p, const uint8_t* end) {
    TensorInfo ti;
    ti.name = read_str(p, end);

    if (p + 4 > end) return nullptr;
    uint32_t n_dims = rd<uint32_t>(p);
    if (n_dims > 8 || p + (uint64_t)n_dims * 8 > end) return nullptr;  // shape dims
    for (uint32_t d = 0; d < n_dims; ++d)
        ti.shape.push_back(rd<uint64_t>(p));

    if (p + 12 > end) return nullptr;   // 4 (type) + 8 (offset)
    ti.type   = (GGMLType)rd<uint32_t>(p);
    ti.offset = rd<uint64_t>(p);

    std::string name = ti.name;
    tensors_.emplace(std::move(name), std::move(ti));
    return p;
}

// ══════════════════════════════════════════════════════════════════════════
//  Accesseurs métadonnées
// ══════════════════════════════════════════════════════════════════════════
bool GGUFParser::has_meta(const std::string& key) const {
    return meta_.count(key) > 0;
}

const MetaValue& GGUFParser::meta(const std::string& key) const {
    static MetaValue empty;
    auto it = meta_.find(key);
    return it != meta_.end() ? it->second : empty;
}

uint32_t GGUFParser::meta_u32(const std::string& key, uint32_t def) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return def;
    switch (it->second.kind) {
        case MetaValue::U32: return it->second.u32;
        case MetaValue::I32: return (uint32_t)it->second.i32;
        case MetaValue::U64: return (uint32_t)it->second.u64;
        default:             return def;
    }
}

int32_t GGUFParser::meta_i32(const std::string& key, int32_t def) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return def;
    switch (it->second.kind) {
        case MetaValue::I32: return it->second.i32;
        case MetaValue::U32: return (int32_t)it->second.u32;
        default:             return def;
    }
}

uint64_t GGUFParser::meta_u64(const std::string& key, uint64_t def) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return def;
    switch (it->second.kind) {
        case MetaValue::U64: return it->second.u64;
        case MetaValue::U32: return it->second.u32;
        default:             return def;
    }
}

float GGUFParser::meta_f32(const std::string& key, float def) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return def;
    switch (it->second.kind) {
        case MetaValue::F32: return it->second.f32;
        case MetaValue::F64: return (float)it->second.f64;
        default:             return def;
    }
}

std::string GGUFParser::meta_str(const std::string& key, const std::string& def) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return def;
    return (it->second.kind == MetaValue::STR) ? it->second.str : def;
}

const std::vector<std::string>& GGUFParser::meta_arr_str(const std::string& key) const {
    static std::vector<std::string> empty;
    auto it = meta_.find(key);
    if (it == meta_.end() || it->second.kind != MetaValue::ARR_STR) return empty;
    return it->second.arr_str;
}

// ══════════════════════════════════════════════════════════════════════════
//  Accesseurs tenseurs
// ══════════════════════════════════════════════════════════════════════════
bool GGUFParser::has_tensor(const std::string& name) const {
    return tensors_.count(name) > 0;
}

const TensorInfo& GGUFParser::tensor_info(const std::string& name) const {
    static TensorInfo empty;
    auto it = tensors_.find(name);
    return it != tensors_.end() ? it->second : empty;
}

const uint8_t* GGUFParser::tensor_raw(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return data_base_ + it->second.offset;
}

std::vector<std::string> GGUFParser::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (auto& [k, _] : tensors_) out.push_back(k);
    std::sort(out.begin(), out.end());
    return out;
}

void GGUFParser::dequantize_into(const std::string& name, std::vector<float>& out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return;
    const TensorInfo& ti = it->second;
    const uint64_t n = ti.n_elems();
    if (n == 0) return;
    if (out.size() < n) out.resize(n);  // grandit si besoin, ne rétrécit jamais
    const uint8_t* src = data_base_ + ti.offset;
    switch (ti.type) {
        case GGMLType::F32:  dq_f32 (src, n, out.data()); break;
        case GGMLType::F16:  dq_f16 (src, n, out.data()); break;
        case GGMLType::BF16: dq_bf16(src, n, out.data()); break;
        case GGMLType::Q4_0: dq_q4_0(src, n, out.data()); break;
        case GGMLType::Q8_0: dq_q8_0(src, n, out.data()); break;
        case GGMLType::Q4_K: dq_q4_k(src, n, out.data()); break;
        case GGMLType::Q6_K: dq_q6_k(src, n, out.data()); break;
        default:
            LOG_WARN("GGUFParser", "Dequantisation non implémentée pour type "
                     + std::to_string((uint32_t)ti.type));
            break;
    }
}

void GGUFParser::dequantize_row(const std::string& name, int64_t row, float* out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !out) return;
    const TensorInfo& ti = it->second;

    // shape[0] = cols (dim la plus rapide dans GGML)
    const uint64_t cols = ti.shape.empty() ? 0 : ti.shape[0];
    const uint8_t* base = data_base_ + ti.offset;

    switch (ti.type) {
    case GGMLType::F32:
        std::memcpy(out, (const float*)base + row * cols, cols * 4);
        break;
    case GGMLType::F16:
        dq_f16(base + (uint64_t)row * cols * 2, cols, out);
        break;
    case GGMLType::BF16:
        dq_bf16(base + (uint64_t)row * cols * 2, cols, out);
        break;
    case GGMLType::Q4_K:
        dq_q4_k(base + (uint64_t)row * (cols / 256) * 144, cols, out);
        break;
    case GGMLType::Q6_K:
        dq_q6_k(base + (uint64_t)row * (cols / 256) * 210, cols, out);
        break;
    case GGMLType::Q8_0:
        dq_q8_0(base + (uint64_t)row * (cols / 32) * 34, cols, out);
        break;
    case GGMLType::Q4_0:
        dq_q4_0(base + (uint64_t)row * (cols / 32) * 18, cols, out);
        break;
    default: {
        std::vector<float> full;
        dequantize_into(name, full);
        if (full.empty()) std::memset(out, 0, cols * 4);
        else std::memcpy(out, full.data() + row * cols, cols * 4);
        break;
    }
    }
}

std::vector<float> GGUFParser::dequantize(const std::string& name) const {
    std::vector<float> out;
    dequantize_into(name, out);
    return out;
}

// ══════════════════════════════════════════════════════════════════════════
//  GEMV quantisé — dot(x, W_row_j) sans expansion float32
//  Élimine 11 Go d'écritures/lectures de wbuf_ par passe transformer.
// ══════════════════════════════════════════════════════════════════════════
//  dot_q8_0 : Q8_0 row × float vector  (bloc 34 octets : f16 d + 32×int8)
// ══════════════════════════════════════════════════════════════════════════
float GGUFParser::dot_q8_0(const uint8_t* row, int n_cols, const float* x) {
    const int nb = n_cols / 32;
#ifdef __AVX2__
    __m256 vacc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + b * 34;
        const float    d   = fp16(*(const uint16_t*)blk);
        const int8_t*  qs  = (const int8_t*)(blk + 2);
        const float*   xb  = x + b * 32;
        const __m256   vd  = _mm256_set1_ps(d);
        // 4 groupes de 8 int8 → float
        for (int g = 0; g < 4; g++) {
            const __m128i i8v = _mm_loadl_epi64((const __m128i*)(qs + g * 8));
            const __m256  f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8v));
            const __m256  vx  = _mm256_loadu_ps(xb + g * 8);
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(vd, f32), vx, vacc);
        }
    }
    const __m128 lo   = _mm256_castps256_ps128(vacc);
    const __m128 hi   = _mm256_extractf128_ps(vacc, 1);
    const __m128 s128 = _mm_add_ps(lo, hi);
    const __m128 s64  = _mm_add_ps(s128, _mm_movehl_ps(s128, s128));
    const __m128 s32  = _mm_add_ss(s64, _mm_shuffle_ps(s64, s64, 1));
    return _mm_cvtss_f32(s32);
#elif defined(__ARM_NEON)
    float32x4_t vacc = vdupq_n_f32(0.f);
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + b * 34;
        const float    d   = fp16(*(const uint16_t*)blk);
        const int8_t*  qs  = (const int8_t*)(blk + 2);
        const float*   xb  = x + b * 32;
        const float32x4_t vd = vdupq_n_f32(d);
        for (int g = 0; g < 4; g++) {
            const int8x8_t i8v = vld1_s8(qs + g * 8);
            const int16x8_t i16v = vmovl_s8(i8v);
            const float32x4_t flo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16v)));
            const float32x4_t fhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16v)));
            vacc = vfmaq_f32(vacc, vmulq_f32(vd, flo), vld1q_f32(xb + g * 8));
            vacc = vfmaq_f32(vacc, vmulq_f32(vd, fhi), vld1q_f32(xb + g * 8 + 4));
        }
    }
    return hsum_neon(vacc);
#else
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + b * 34;
        const float    d   = fp16(*(const uint16_t*)blk);
        const int8_t*  qs  = (const int8_t*)(blk + 2);
        const float*   xb  = x + b * 32;
        float s = 0.f;
        for (int l = 0; l < 32; l++) s += (float)qs[l] * xb[l];
        acc += d * s;
    }
    return acc;
#endif
}

// ══════════════════════════════════════════════════════════════════════════
//  dot_q4_0 : Q4_0 row × float vector  (bloc 18 octets : f16 d + 16×uint8)
// ══════════════════════════════════════════════════════════════════════════
float GGUFParser::dot_q4_0(const uint8_t* row, int n_cols, const float* x) {
    const int nb = n_cols / 32;
#ifdef __ARM_NEON
    float32x4_t vacc = vdupq_n_f32(0.f);
    const uint8x8_t mask4 = vdup_n_u8(0x0F);
    const int8x8_t  voff  = vdup_n_s8(8);
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + b * 18;
        const float    d   = fp16(*(const uint16_t*)blk);
        const uint8_t* qs  = blk + 2;
        const float*   xb  = x + b * 32;
        const float32x4_t vd = vdupq_n_f32(d);
        for (int l = 0; l < 16; l += 8) {
            const uint8x8_t raw = vld1_u8(qs + l);
            const uint8x8_t lo = vand_u8(raw, mask4);
            const uint8x8_t hi = vshr_n_u8(raw, 4);
            const int8x8_t lo_s = vsub_s8(vreinterpret_s8_u8(lo), voff);
            const int8x8_t hi_s = vsub_s8(vreinterpret_s8_u8(hi), voff);
            const int16x8_t lo16 = vmovl_s8(lo_s);
            const float32x4_t flo_a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            const float32x4_t flo_b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
            vacc = vfmaq_f32(vacc, vmulq_f32(vd, flo_a), vld1q_f32(xb + l));
            vacc = vfmaq_f32(vacc, vmulq_f32(vd, flo_b), vld1q_f32(xb + l + 4));
            const int16x8_t hi16 = vmovl_s8(hi_s);
            const float32x4_t fhi_a = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            const float32x4_t fhi_b = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
            vacc = vfmaq_f32(vacc, vmulq_f32(vd, fhi_a), vld1q_f32(xb + l + 16));
            vacc = vfmaq_f32(vacc, vmulq_f32(vd, fhi_b), vld1q_f32(xb + l + 20));
        }
    }
    return hsum_neon(vacc);
#elif defined(__AVX2__)
    __m256 vacc = _mm256_setzero_ps();
    const __m256i m0F = _mm256_set1_epi32(0x0F);
    const __m256i v8  = _mm256_set1_epi32(8);
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + b * 18;
        const float    d   = fp16(*(const uint16_t*)blk);
        const uint8_t* qs  = blk + 2;
        const float*   xb  = x + b * 32;
        __m256 sl = _mm256_setzero_ps(), sh = _mm256_setzero_ps();
        for (int l = 0; l < 16; l += 8) {
            const __m256i q  = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qs + l)));
            // low nibble  → x[l..],  high nibble → x[l+16..], chacun décalé de -8
            const __m256 lo = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(q, m0F), v8));
            const __m256 hi = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(q, 4), v8));
            sl = _mm256_fmadd_ps(lo, _mm256_loadu_ps(xb + l),      sl);
            sh = _mm256_fmadd_ps(hi, _mm256_loadu_ps(xb + l + 16), sh);
        }
        // acc += d * (sl + sh) — le scale du bloc appliqué une fois.
        vacc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_add_ps(sl, sh), vacc);
    }
    {
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    }
#else
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = row + b * 18;
        const float    d   = fp16(*(const uint16_t*)blk);
        const uint8_t* qs  = blk + 2;
        const float*   xb  = x + b * 32;
        float s = 0.f;
        for (int l = 0; l < 16; l++) {
            s += (float)((int)(qs[l] & 0xF) - 8) * xb[l];
            s += (float)((int)(qs[l] >> 4)  - 8) * xb[l + 16];
        }
        acc += d * s;
    }
    return acc;
#endif
}

// ══════════════════════════════════════════════════════════════════════════

float GGUFParser::dot_q4k(const uint8_t* blk_data, int n_cols, const float* x) {
    float acc = 0.f;
    const int nb = n_cols / 256;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk    = blk_data + b * 144;
        const float d    = fp16(*(const uint16_t*)(blk + 0));
        const float dmin = fp16(*(const uint16_t*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs     = blk + 16;
        const float*   xb     = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            scale_min_k4(is,     scales, sc0, m0);
            scale_min_k4(is + 1, scales, sc1, m1);
            const float d1 = d * sc0, m1f = dmin * m0;
            const float d2 = d * sc1, m2f = dmin * m1;
            const float* xs = xb + j;
            float s1 = 0.f, s2 = 0.f, sx1 = 0.f, sx2 = 0.f;
#ifdef __AVX2__
            __m256 va1 = _mm256_setzero_ps(), va2 = _mm256_setzero_ps();
            __m256 vb1 = _mm256_setzero_ps(), vb2 = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 8) {
                __m128i vq    = _mm_loadl_epi64((const __m128i*)(qs + l));
                __m256i vq32  = _mm256_cvtepu8_epi32(vq);
                __m256  vflo  = _mm256_cvtepi32_ps(_mm256_and_si256(vq32, _mm256_set1_epi32(0xF)));
                __m256  vfhi  = _mm256_cvtepi32_ps(_mm256_srli_epi32(vq32, 4));
                __m256  vx1   = _mm256_loadu_ps(xs + l);
                __m256  vx2   = _mm256_loadu_ps(xs + l + 32);
                va1 = _mm256_fmadd_ps(vflo, vx1, va1);
                va2 = _mm256_fmadd_ps(vfhi, vx2, va2);
                vb1 = _mm256_add_ps(vb1, vx1);
                vb2 = _mm256_add_ps(vb2, vx2);
            }
#define HSUM8(v, r) { \
    __m128 _lo = _mm256_castps256_ps128(v); \
    __m128 _hi = _mm256_extractf128_ps(v,1); \
    __m128 _s  = _mm_add_ps(_lo,_hi); \
    _s = _mm_hadd_ps(_s,_s); _s = _mm_hadd_ps(_s,_s); r = _mm_cvtss_f32(_s); }
            HSUM8(va1,s1); HSUM8(va2,s2); HSUM8(vb1,sx1); HSUM8(vb2,sx2);
#undef HSUM8
#elif defined(__ARM_NEON)
            float32x4_t va1 = vdupq_n_f32(0.f), va2 = vdupq_n_f32(0.f);
            float32x4_t vb1 = vdupq_n_f32(0.f), vb2 = vdupq_n_f32(0.f);
            const uint8x8_t vmask4 = vdup_n_u8(0x0F);
            for (int l = 0; l < 32; l += 8) {
                const uint8x8_t vq = vld1_u8(qs + l);
                const uint8x8_t lo = vand_u8(vq, vmask4);
                const uint8x8_t hi = vshr_n_u8(vq, 4);
                const uint16x8_t lo16 = vmovl_u8(lo);
                const float32x4_t flo_a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
                const float32x4_t flo_b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
                const uint16x8_t hi16 = vmovl_u8(hi);
                const float32x4_t fhi_a = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
                const float32x4_t fhi_b = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
                const float32x4_t vx1a = vld1q_f32(xs + l);
                const float32x4_t vx1b = vld1q_f32(xs + l + 4);
                const float32x4_t vx2a = vld1q_f32(xs + l + 32);
                const float32x4_t vx2b = vld1q_f32(xs + l + 36);
                va1 = vfmaq_f32(va1, flo_a, vx1a);
                va1 = vfmaq_f32(va1, flo_b, vx1b);
                va2 = vfmaq_f32(va2, fhi_a, vx2a);
                va2 = vfmaq_f32(va2, fhi_b, vx2b);
                vb1 = vaddq_f32(vb1, vx1a);
                vb1 = vaddq_f32(vb1, vx1b);
                vb2 = vaddq_f32(vb2, vx2a);
                vb2 = vaddq_f32(vb2, vx2b);
            }
            s1 = hsum_neon(va1); s2 = hsum_neon(va2);
            sx1 = hsum_neon(vb1); sx2 = hsum_neon(vb2);
#else
            for (int l = 0; l < 32; l++) {
                s1  += (float)(qs[l] & 0xF) * xs[l];
                s2  += (float)(qs[l] >> 4)  * xs[l + 32];
                sx1 += xs[l];
                sx2 += xs[l + 32];
            }
#endif
            acc += d1 * s1 - m1f * sx1 + d2 * s2 - m2f * sx2;
            qs += 32;
            is += 2;
        }
    }
    return acc;
}

float GGUFParser::dot_q6k(const uint8_t* blk_data, int n_cols, const float* x) {
    const int nb = n_cols / 256;
#ifdef __ARM_NEON
    float32x4_t vtotal = vdupq_n_f32(0.f);
    const uint8x8_t mask4 = vdup_n_u8(0x0F);
    const uint8x8_t mask2 = vdup_n_u8(0x03);
    const int8x8_t  voff  = vdup_n_s8(32);
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = blk_data + b * 210;
        const uint8_t* ql  = blk;
        const uint8_t* qh  = blk + 128;
        const int8_t*  sc  = (const int8_t*)(blk + 192);
        const float    d   = fp16(*(const uint16_t*)(blk + 208));
        const float*   xb  = x + b * 256;
        for (int j = 0; j < 256; j += 128) {
            const float* xs = xb + j;
            float32x4_t va0 = vdupq_n_f32(0.f), va1 = vdupq_n_f32(0.f);
            float32x4_t va2 = vdupq_n_f32(0.f), va3 = vdupq_n_f32(0.f);
            for (int l = 0; l < 32; l += 8) {
                const int is = l / 16;   // Q6_K : scale par sous-bloc de 16 (correction)
                const float32x4_t vdsc0 = vdupq_n_f32(d * sc[is + 0]);
                const float32x4_t vdsc2 = vdupq_n_f32(d * sc[is + 2]);
                const float32x4_t vdsc4 = vdupq_n_f32(d * sc[is + 4]);
                const float32x4_t vdsc6 = vdupq_n_f32(d * sc[is + 6]);
                const uint8x8_t vql0 = vld1_u8(ql + l);
                const uint8x8_t vql1 = vld1_u8(ql + l + 32);
                const uint8x8_t vqh  = vld1_u8(qh + l);
                const uint8x8_t q0 = vorr_u8(vand_u8(vql0, mask4),
                                     vshl_n_u8(vand_u8(vqh, mask2), 4));
                const uint8x8_t q1 = vorr_u8(vand_u8(vql1, mask4),
                                     vshl_n_u8(vand_u8(vshr_n_u8(vqh, 2), mask2), 4));
                const uint8x8_t q2 = vorr_u8(vshr_n_u8(vql0, 4),
                                     vshl_n_u8(vand_u8(vshr_n_u8(vqh, 4), mask2), 4));
                const uint8x8_t q3 = vorr_u8(vshr_n_u8(vql1, 4),
                                     vshl_n_u8(vshr_n_u8(vqh, 6), 4));
                const int8x8_t sq0 = vsub_s8(vreinterpret_s8_u8(q0), voff);
                const int8x8_t sq1 = vsub_s8(vreinterpret_s8_u8(q1), voff);
                const int8x8_t sq2 = vsub_s8(vreinterpret_s8_u8(q2), voff);
                const int8x8_t sq3 = vsub_s8(vreinterpret_s8_u8(q3), voff);
                int16x8_t i16;
                float32x4_t fa, fb;
                i16 = vmovl_s8(sq0);
                fa = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                fb = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                va0 = vfmaq_f32(va0, vmulq_f32(vdsc0, fa), vld1q_f32(xs + l));
                va0 = vfmaq_f32(va0, vmulq_f32(vdsc0, fb), vld1q_f32(xs + l + 4));
                i16 = vmovl_s8(sq1);
                fa = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                fb = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                va1 = vfmaq_f32(va1, vmulq_f32(vdsc2, fa), vld1q_f32(xs + l + 32));
                va1 = vfmaq_f32(va1, vmulq_f32(vdsc2, fb), vld1q_f32(xs + l + 36));
                i16 = vmovl_s8(sq2);
                fa = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                fb = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                va2 = vfmaq_f32(va2, vmulq_f32(vdsc4, fa), vld1q_f32(xs + l + 64));
                va2 = vfmaq_f32(va2, vmulq_f32(vdsc4, fb), vld1q_f32(xs + l + 68));
                i16 = vmovl_s8(sq3);
                fa = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16)));
                fb = vcvtq_f32_s32(vmovl_s16(vget_high_s16(i16)));
                va3 = vfmaq_f32(va3, vmulq_f32(vdsc6, fa), vld1q_f32(xs + l + 96));
                va3 = vfmaq_f32(va3, vmulq_f32(vdsc6, fb), vld1q_f32(xs + l + 100));
            }
            vtotal = vaddq_f32(vtotal, vaddq_f32(vaddq_f32(va0, va1), vaddq_f32(va2, va3)));
            ql += 64;
            qh += 32;
            sc +=  8;
        }
    }
    return hsum_neon(vtotal);
#elif defined(__AVX2__)
    __m256 va0 = _mm256_setzero_ps(), va1 = _mm256_setzero_ps();
    __m256 va2 = _mm256_setzero_ps(), va3 = _mm256_setzero_ps();
    const __m256i m0F = _mm256_set1_epi32(0x0F);
    const __m256i m03 = _mm256_set1_epi32(0x03);
    const __m256i v32 = _mm256_set1_epi32(32);
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = blk_data + b * 210;
        const uint8_t* ql  = blk;
        const uint8_t* qh  = blk + 128;
        const int8_t*  sc  = (const int8_t*)(blk + 192);
        const float    d   = fp16(*(const uint16_t*)(blk + 208));
        const float*   xb  = x + b * 256;
        for (int j = 0; j < 256; j += 128) {
            const float* xs = xb + j;
            for (int l = 0; l < 32; l += 8) {
                const int is = l / 16;   // Q6_K : scale par sous-bloc de 16
                const __m256 vdsc0 = _mm256_set1_ps(d * sc[is + 0]);
                const __m256 vdsc2 = _mm256_set1_ps(d * sc[is + 2]);
                const __m256 vdsc4 = _mm256_set1_ps(d * sc[is + 4]);
                const __m256 vdsc6 = _mm256_set1_ps(d * sc[is + 6]);
                const __m256i vql0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + l)));
                const __m256i vql1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ql + l + 32)));
                const __m256i vqh  = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qh + l)));
                // q = (nibble | (bits_hauts<<4)) - 32, exactement comme le scalaire.
                const __m256i q0 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(vql0, m0F),
                    _mm256_slli_epi32(_mm256_and_si256(vqh, m03), 4)), v32);
                const __m256i q1 = _mm256_sub_epi32(_mm256_or_si256(_mm256_and_si256(vql1, m0F),
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(vqh, 2), m03), 4)), v32);
                const __m256i q2 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(vql0, 4),
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(vqh, 4), m03), 4)), v32);
                const __m256i q3 = _mm256_sub_epi32(_mm256_or_si256(_mm256_srli_epi32(vql1, 4),
                    _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(vqh, 6), m03), 4)), v32);
                va0 = _mm256_fmadd_ps(_mm256_mul_ps(vdsc0, _mm256_cvtepi32_ps(q0)), _mm256_loadu_ps(xs + l),      va0);
                va1 = _mm256_fmadd_ps(_mm256_mul_ps(vdsc2, _mm256_cvtepi32_ps(q1)), _mm256_loadu_ps(xs + l + 32), va1);
                va2 = _mm256_fmadd_ps(_mm256_mul_ps(vdsc4, _mm256_cvtepi32_ps(q2)), _mm256_loadu_ps(xs + l + 64), va2);
                va3 = _mm256_fmadd_ps(_mm256_mul_ps(vdsc6, _mm256_cvtepi32_ps(q3)), _mm256_loadu_ps(xs + l + 96), va3);
            }
            ql += 64;
            qh += 32;
            sc +=  8;
        }
    }
    {
        __m256 vs = _mm256_add_ps(_mm256_add_ps(va0, va1), _mm256_add_ps(va2, va3));
        __m128 lo = _mm256_castps256_ps128(vs);
        __m128 hi = _mm256_extractf128_ps(vs, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    }
#else
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        const uint8_t* blk = blk_data + b * 210;
        const uint8_t* ql  = blk;
        const uint8_t* qh  = blk + 128;
        const int8_t*  sc  = (const int8_t*)(blk + 192);
        const float    d   = fp16(*(const uint16_t*)(blk + 208));
        const float*   xb  = x + b * 256;
        for (int j = 0; j < 256; j += 128) {
            const float* xs = xb + j;
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;   // Q6_K : scale par sous-bloc de 16 (correction)
                acc += d * sc[is + 0] * (float)((int8_t)((ql[l+ 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32) * xs[l +  0];
                acc += d * sc[is + 2] * (float)((int8_t)((ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32) * xs[l + 32];
                acc += d * sc[is + 4] * (float)((int8_t)((ql[l+ 0] >>  4)  | (((qh[l] >> 4) & 3) << 4)) - 32) * xs[l + 64];
                acc += d * sc[is + 6] * (float)((int8_t)((ql[l+32] >>  4)  | (((qh[l] >> 6) & 3) << 4)) - 32) * xs[l + 96];
            }
            ql += 64;
            qh += 32;
            sc +=  8;
        }
    }
    return acc;
#endif
}

bool GGUFParser::gemv_q(const std::string& name, float* out, const float* x,
                         int n_out, int n_in) const {
    return gemv_q_range(name, out, x, 0, n_out, n_in);
}

bool GGUFParser::gemv_q_range(const std::string& name, float* out, const float* x,
                               int j0, int j1, int n_in) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;
    const TensorInfo& ti = it->second;
    const uint8_t* base = data_base_ + ti.offset;
    // Prefetch de la ligne de poids suivante pendant le calcul de la courante,
    // pour masquer la latence RAM au franchissement de la frontière de ligne.
    #define ORACLE_PF_NEXT_ROW \
        if (j + 1 < j1) { \
            const uint8_t* nxt = base + (uint64_t)(j + 1) * row_bytes; \
            __builtin_prefetch(nxt, 0, 3); \
            __builtin_prefetch(nxt + 64, 0, 3); \
        }
    switch (ti.type) {
    case GGMLType::Q8_0: {
        const uint64_t row_bytes = (uint64_t)(n_in / 32) * 34;
        #pragma omp parallel for schedule(static)
        for (int j = j0; j < j1; j++) {
            ORACLE_PF_NEXT_ROW
            out[j] = dot_q8_0(base + (uint64_t)j * row_bytes, n_in, x);
        }
        return true;
    }
    case GGMLType::Q4_0: {
        const uint64_t row_bytes = (uint64_t)(n_in / 32) * 18;
        #pragma omp parallel for schedule(static)
        for (int j = j0; j < j1; j++) {
            ORACLE_PF_NEXT_ROW
            out[j] = dot_q4_0(base + (uint64_t)j * row_bytes, n_in, x);
        }
        return true;
    }
    case GGMLType::Q4_K: {
        const uint64_t row_bytes = (uint64_t)(n_in / 256) * 144;
        #pragma omp parallel for schedule(static)
        for (int j = j0; j < j1; j++) {
            ORACLE_PF_NEXT_ROW
            out[j] = dot_q4k(base + (uint64_t)j * row_bytes, n_in, x);
        }
        return true;
    }
    case GGMLType::Q6_K: {
        const uint64_t row_bytes = (uint64_t)(n_in / 256) * 210;
        #pragma omp parallel for schedule(static)
        for (int j = j0; j < j1; j++) {
            ORACLE_PF_NEXT_ROW
            out[j] = dot_q6k(base + (uint64_t)j * row_bytes, n_in, x);
        }
        return true;
    }
    default:
        return false;
    }
    #undef ORACLE_PF_NEXT_ROW
}

// ══════════════════════════════════════════════════════════════════════════
//  GEMM quantisé "weight-stationary" — pour le prefill batché
//  out[n_tokens × n_out] = X[n_tokens × n_in] @ W[n_out × n_in]^T
//  Chaque ligne de poids est décodée UNE fois (dequantize_row) puis appliquée
//  à tous les tokens → le coût du dequant est amorti sur tout le batch.
// ══════════════════════════════════════════════════════════════════════════
bool GGUFParser::gemm_q(const std::string& name, float* out, const float* x,
                        int n_tokens, int n_out, int n_in) const {
    if (tensors_.find(name) == tensors_.end()) return false;
    #pragma omp parallel
    {
        std::vector<float> row(n_in);
        #pragma omp for schedule(static)
        for (int j = 0; j < n_out; j++) {
            dequantize_row(name, j, row.data());   // décode la ligne UNE fois
            const float* w = row.data();

            // Micro-kernel tuilé sur la dimension token : 4 tokens par passe.
            // La ligne de poids n'est chargée qu'UNE fois pour 4 produits
            // scalaires (au lieu d'une fois par token), et les 8 chaînes FMA
            // indépendantes (4 tokens × 2 accumulateurs) saturent les unités
            // FMA en masquant leur latence.
            int t = 0;
            for (; t + 4 <= n_tokens; t += 4) {
                const float* x0 = x + (size_t)(t + 0) * n_in;
                const float* x1 = x + (size_t)(t + 1) * n_in;
                const float* x2 = x + (size_t)(t + 2) * n_in;
                const float* x3 = x + (size_t)(t + 3) * n_in;
                float r0, r1, r2, r3;
#ifdef __AVX2__
                __m256 a0=_mm256_setzero_ps(), b0=_mm256_setzero_ps();
                __m256 a1=_mm256_setzero_ps(), b1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(), b2=_mm256_setzero_ps();
                __m256 a3=_mm256_setzero_ps(), b3=_mm256_setzero_ps();
                int k = 0;
                for (; k + 16 <= n_in; k += 16) {
                    __m256 w0 = _mm256_loadu_ps(w + k);
                    __m256 w1 = _mm256_loadu_ps(w + k + 8);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0+k),   w0, a0);
                    b0 = _mm256_fmadd_ps(_mm256_loadu_ps(x0+k+8), w1, b0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1+k),   w0, a1);
                    b1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1+k+8), w1, b1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2+k),   w0, a2);
                    b2 = _mm256_fmadd_ps(_mm256_loadu_ps(x2+k+8), w1, b2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3+k),   w0, a3);
                    b3 = _mm256_fmadd_ps(_mm256_loadu_ps(x3+k+8), w1, b3);
                }
                r0 = hsum256(_mm256_add_ps(a0, b0));
                r1 = hsum256(_mm256_add_ps(a1, b1));
                r2 = hsum256(_mm256_add_ps(a2, b2));
                r3 = hsum256(_mm256_add_ps(a3, b3));
                for (; k < n_in; k++) {
                    float wk = w[k];
                    r0 += x0[k]*wk; r1 += x1[k]*wk; r2 += x2[k]*wk; r3 += x3[k]*wk;
                }
#elif defined(__ARM_NEON)
                float32x4_t a0=vdupq_n_f32(0.f),a1=vdupq_n_f32(0.f),a2=vdupq_n_f32(0.f),a3=vdupq_n_f32(0.f);
                float32x4_t b0=vdupq_n_f32(0.f),b1=vdupq_n_f32(0.f),b2=vdupq_n_f32(0.f),b3=vdupq_n_f32(0.f);
                int k = 0;
                for (; k + 8 <= n_in; k += 8) {
                    float32x4_t w0=vld1q_f32(w+k), w1=vld1q_f32(w+k+4);
                    a0=vfmaq_f32(a0, vld1q_f32(x0+k),   w0); b0=vfmaq_f32(b0, vld1q_f32(x0+k+4), w1);
                    a1=vfmaq_f32(a1, vld1q_f32(x1+k),   w0); b1=vfmaq_f32(b1, vld1q_f32(x1+k+4), w1);
                    a2=vfmaq_f32(a2, vld1q_f32(x2+k),   w0); b2=vfmaq_f32(b2, vld1q_f32(x2+k+4), w1);
                    a3=vfmaq_f32(a3, vld1q_f32(x3+k),   w0); b3=vfmaq_f32(b3, vld1q_f32(x3+k+4), w1);
                }
                r0=hsum_neon(vaddq_f32(a0,b0)); r1=hsum_neon(vaddq_f32(a1,b1));
                r2=hsum_neon(vaddq_f32(a2,b2)); r3=hsum_neon(vaddq_f32(a3,b3));
                for (; k < n_in; k++){ float wk=w[k]; r0+=x0[k]*wk; r1+=x1[k]*wk; r2+=x2[k]*wk; r3+=x3[k]*wk; }
#else
                r0=r1=r2=r3=0.f;
                for (int k=0;k<n_in;k++){ float wk=w[k]; r0+=x0[k]*wk; r1+=x1[k]*wk; r2+=x2[k]*wk; r3+=x3[k]*wk; }
#endif
                out[(size_t)(t+0)*n_out + j] = r0;
                out[(size_t)(t+1)*n_out + j] = r1;
                out[(size_t)(t+2)*n_out + j] = r2;
                out[(size_t)(t+3)*n_out + j] = r3;
            }
            // Tokens restants (n_tokens % 4) : produit scalaire simple.
            for (; t < n_tokens; t++)
                out[(size_t)t * n_out + j] = ops::dot(x + (size_t)t * n_in, w, n_in);
        }
    }
    return true;
}

// ══════════════════════════════════════════════════════════════════════════
//  Conversion FP16 → float32
// ══════════════════════════════════════════════════════════════════════════
float GGUFParser::fp16(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant =  h        & 0x3FFu;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            uint32_t e = 0, m = mant;
            while (!(m & 0x400)) { m <<= 1; e++; }
            f = (sign << 31) | ((127u - 15u - e + 1u) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float r; std::memcpy(&r, &f, 4); return r;
}

// ══════════════════════════════════════════════════════════════════════════
//  Dequantisation
// ══════════════════════════════════════════════════════════════════════════

// F32 — copie directe
void GGUFParser::dq_f32(const uint8_t* s, uint64_t n, float* out) {
    std::memcpy(out, s, n * 4);
}

// BF16 → F32
void GGUFParser::dq_bf16(const uint8_t* s, uint64_t n, float* out) {
    const uint16_t* src = (const uint16_t*)s;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t f = (uint32_t)src[i] << 16;
        std::memcpy(&out[i], &f, 4);
    }
}

// F16 → F32
void GGUFParser::dq_f16(const uint8_t* s, uint64_t n, float* out) {
    const uint16_t* src = (const uint16_t*)s;
#ifdef __ARM_NEON
    uint64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float16x4_t vh = vld1_f16((const float16_t*)(src + i));
        vst1q_f32(out + i, vcvt_f32_f16(vh));
    }
    for (; i < n; ++i) out[i] = fp16(src[i]);
#else
    for (uint64_t i = 0; i < n; ++i) out[i] = fp16(src[i]);
#endif
}

// Q4_0 : block de 32 éléments = f16 delta (2) + 16 octets nibbles (16) = 18 octets
void GGUFParser::dq_q4_0(const uint8_t* s, uint64_t n, float* out) {
    const uint64_t nb = n / 32;
    for (uint64_t b = 0; b < nb; ++b) {
        const uint8_t* blk = s + b * 18;
        const float    d   = fp16(*(const uint16_t*)blk);
        const uint8_t* qs  = blk + 2;
        float*         dst = out + b * 32;
        for (int l = 0; l < 16; ++l) {
            dst[l]      = d * (float)((int)(qs[l] & 0xF) - 8);
            dst[l + 16] = d * (float)((int)(qs[l] >> 4)  - 8);
        }
    }
}

// Q8_0 : block de 32 éléments = f16 delta (2) + 32 int8 = 34 octets
void GGUFParser::dq_q8_0(const uint8_t* s, uint64_t n, float* out) {
    const uint64_t nb = n / 32;
    for (uint64_t b = 0; b < nb; ++b) {
        const uint8_t* blk = s + b * 34;
        const float    d   = fp16(*(const uint16_t*)blk);
        const int8_t*  qs  = (const int8_t*)(blk + 2);
        float*         dst = out + b * 32;
        for (int l = 0; l < 32; ++l) dst[l] = d * (float)qs[l];
    }
}

// ── Q4_K — helper : décode scale et min pour le sous-bloc j ──────────────
void GGUFParser::scale_min_k4(int j, const uint8_t* sc, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = sc[j]   & 63;
        m = sc[j+4] & 63;
    } else {
        d = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
        m = (sc[j+4] >> 4)   | ((sc[j-0] >> 6) << 4);
    }
}

// Q4_K : block de 256 éléments = f16 d (2) + f16 dmin (2) + scales[12] + qs[128] = 144 octets
void GGUFParser::dq_q4_k(const uint8_t* s, uint64_t n, float* out) {
    const uint64_t nb = n / 256;
    for (uint64_t b = 0; b < nb; ++b) {
        const uint8_t* blk    = s + b * 144;
        const float    d      = fp16(*(const uint16_t*)(blk + 0));
        const float    dmin   = fp16(*(const uint16_t*)(blk + 2));
        const uint8_t* scales = blk + 4;
        const uint8_t* qs     = blk + 16;
        float*         dst    = out + b * 256;

        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            scale_min_k4(is,     scales, sc0, m0);
            scale_min_k4(is + 1, scales, sc1, m1);
            const float d1 = d * sc0, m1f = dmin * m0;
            const float d2 = d * sc1, m2f = dmin * m1;
            for (int l = 0; l < 32; ++l) {
                dst[j + l]      = d1 * (float)(qs[l] & 0xF) - m1f;
                dst[j + 32 + l] = d2 * (float)(qs[l] >> 4)  - m2f;
            }
            qs += 32;
            is += 2;
        }
    }
}

// Q6_K : block de 256 éléments = ql[128] + qh[64] + scales[16] + f16 d[2] = 210 octets
void GGUFParser::dq_q6_k(const uint8_t* s, uint64_t n, float* out) {
    const uint64_t nb = n / 256;
    for (uint64_t b = 0; b < nb; ++b) {
        const uint8_t* blk    = s + b * 210;
        const uint8_t* ql     = blk;
        const uint8_t* qh     = blk + 128;
        const int8_t*  sc     = (const int8_t*)(blk + 192);
        const float    d      = fp16(*(const uint16_t*)(blk + 208));
        float*         dst    = out + b * 256;

        // Deux demi-blocs de 128 éléments chacun.
        // On indexe via dst+j (fixe) et on avance seulement les pointeurs source.
        for (int j = 0; j < 256; j += 128) {
            float* dyy = dst + j;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;   // Q6_K : scale par sous-bloc de 16 (correction)
                dyy[l +  0] = d * sc[is + 0] *
                    (float)((int8_t)((ql[l+ 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
                dyy[l + 32] = d * sc[is + 2] *
                    (float)((int8_t)((ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
                dyy[l + 64] = d * sc[is + 4] *
                    (float)((int8_t)((ql[l+ 0] >>  4)  | (((qh[l] >> 4) & 3) << 4)) - 32);
                dyy[l + 96] = d * sc[is + 6] *
                    (float)((int8_t)((ql[l+32] >>  4)  | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            ql += 64;
            qh += 32;
            sc +=  8;
        }
    }
}
