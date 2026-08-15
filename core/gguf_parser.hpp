#pragma once
#include "core/platform.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// ── Types de quantisation GGML ────────────────────────────────────────────
enum class GGMLType : uint32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
    Q8_K  = 15,
    BF16  = 30,
    UNKNOWN = 0xFFFFFFFF,
};

// ── Valeur de métadonnée ─────────────────────────────────────────────────
struct MetaValue {
    enum Kind {
        U8, I8, U16, I16, U32, I32, F32, BOOL,
        STR, U64, I64, F64, ARR_STR, ARR_I32, ARR_U32, ARR_F32, SKIP
    } kind = SKIP;

    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        bool     b;
        uint64_t u64;
        int64_t  i64;
        double   f64;
    };
    std::string            str;
    std::vector<std::string> arr_str;
    std::vector<int32_t>   arr_i32;
    std::vector<uint32_t>  arr_u32;
    std::vector<float>     arr_f32;

    MetaValue() : u64(0) {}
};

// ── Descripteur d'un tenseur ──────────────────────────────────────────────
struct TensorInfo {
    std::string           name;
    std::vector<uint64_t> shape;   // dimensions (n_dims éléments)
    GGMLType              type;
    uint64_t              offset;  // offset depuis le début de la section data

    uint64_t n_elems() const;
    uint64_t n_bytes() const;
};

// ─────────────────────────────────────────────────────────────────────────
// GGUFParser — lit un fichier GGUF sans aucune dépendance externe
// Usage :
//   GGUFParser p;
//   p.load("model.gguf");
//   uint32_t n_heads = p.meta_u32("llama.attention.head_count");
//   auto weights = p.dequantize("token_embd.weight"); // → float32[]
// ─────────────────────────────────────────────────────────────────────────
class GGUFParser {
public:
    GGUFParser()  = default;
    ~GGUFParser() { close(); }

    GGUFParser(const GGUFParser&)            = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;

    bool load(const std::string& path);
    void close();
    bool is_open() const { return data_ != nullptr; }

    // ── Métadonnées ───────────────────────────────────────────────────────
    bool               has_meta(const std::string& key) const;
    const MetaValue&   meta(const std::string& key) const;
    uint32_t           meta_u32(const std::string& key, uint32_t   def = 0)   const;
    int32_t            meta_i32(const std::string& key, int32_t    def = 0)   const;
    uint64_t           meta_u64(const std::string& key, uint64_t   def = 0)   const;
    float              meta_f32(const std::string& key, float      def = 0.f) const;
    std::string        meta_str(const std::string& key, const std::string& def = "") const;
    const std::vector<std::string>& meta_arr_str(const std::string& key) const;

    // ── Tenseurs ──────────────────────────────────────────────────────────
    bool               has_tensor(const std::string& name) const;
    const TensorInfo&  tensor_info(const std::string& name) const;
    const uint8_t*     tensor_raw(const std::string& name)  const;
    std::vector<float> dequantize(const std::string& name)  const;
    void               dequantize_into(const std::string& name, std::vector<float>& out) const;
    void               dequantize_row(const std::string& name, int64_t row, float* out) const;
    std::vector<std::string> tensor_names() const;

    // GEMV quantisé : out[n_out] = W[n_out × n_in] @ x[n_in]  sans expansion F32
    // Retourne false si le type n'est pas supporté (fallback : dequantize_into + matmul)
    bool gemv_q(const std::string& name, float* out, const float* x, int n_out, int n_in) const;

    // Variante sur une plage de lignes [j0, j1) — thread-safe (lecture seule)
    bool gemv_q_range(const std::string& name, float* out, const float* x,
                      int j0, int j1, int n_in) const;

    // GEMM quantisé "weight-stationary" pour le prefill batché :
    //   out[n_tokens × n_out] = X[n_tokens × n_in] @ W[n_out × n_in]^T
    // Chaque ligne de poids est décodée UNE fois puis appliquée aux n_tokens
    // activations (dequant amorti sur tout le batch). Retourne false si absent.
    bool gemm_q(const std::string& name, float* out, const float* x,
                int n_tokens, int n_out, int n_in) const;

    // ── Infos globales ────────────────────────────────────────────────────
    uint32_t version()      const { return version_; }
    uint64_t n_tensors()    const { return n_tensors_; }
    uint64_t n_meta()       const { return meta_.size(); }
    uint64_t file_size()    const { return (uint64_t)fsize_; }
    bool     is_mmap_only() const { return data_virt_ == nullptr; }

    // Hooks de test (sur bloc synthétique, sans fichier)
    static void  test_dq_q6_k(const uint8_t* s, uint64_t n, float* out) { dq_q6_k(s, n, out); }
    static void  test_dq_q4_k(const uint8_t* s, uint64_t n, float* out) { dq_q4_k(s, n, out); }
    static void  test_dq_q4_0(const uint8_t* s, uint64_t n, float* out) { dq_q4_0(s, n, out); }
    static float test_dot_q6k (const uint8_t* row, int n_cols, const float* x) { return dot_q6k(row, n_cols, x); }
    static float test_dot_q4_0(const uint8_t* row, int n_cols, const float* x) { return dot_q4_0(row, n_cols, x); }

private:
    // Fichier mappé en mémoire (cross-platform via platform.hpp)
    platform::MappedFile mmap_ = {};
    const uint8_t* data_   = nullptr;   // vue mmap (tout le fichier)
    size_t         fsize_  = 0;

    // Section données préchargée en RAM pour éviter les page faults mmap
    void*          data_virt_      = nullptr;  // buffer vmem_alloc (ou nullptr si échec)
    size_t         data_virt_size_ = 0;        // taille du buffer
    const uint8_t* data_base_      = nullptr;  // pointe sur data_virt_ ou data_+data_start_

    uint32_t version_     = 0;
    uint64_t n_tensors_   = 0;
    uint64_t data_start_  = 0;          // offset dans le fichier

    std::unordered_map<std::string, MetaValue>  meta_;
    std::unordered_map<std::string, TensorInfo> tensors_;

    // ── Parsing interne ───────────────────────────────────────────────────
    bool             parse();
    const uint8_t*   read_kv         (const uint8_t* p, const uint8_t* end);
    const uint8_t*   read_tensor_info(const uint8_t* p, const uint8_t* end);
    MetaValue        read_value      (const uint8_t*& p, const uint8_t* end, uint32_t kind);
    static std::string read_str      (const uint8_t*& p, const uint8_t* end);

    // ── Dequantisation (in-place, sans allocation) ────────────────────────
    static float fp16(uint16_t h);
    static void  dq_f32 (const uint8_t* s, uint64_t n, float* out);
    static void  dq_f16 (const uint8_t* s, uint64_t n, float* out);
    static void  dq_bf16(const uint8_t* s, uint64_t n, float* out);
    static void  dq_q4_0(const uint8_t* s, uint64_t n, float* out);
    static void  dq_q8_0(const uint8_t* s, uint64_t n, float* out);
    static void  dq_q4_k(const uint8_t* s, uint64_t n, float* out);
    static void  dq_q6_k(const uint8_t* s, uint64_t n, float* out);

    static void  scale_min_k4(int j, const uint8_t* sc, uint8_t& d, uint8_t& m);

    // GEMV sans expansion : dot(x, W_row_j) pour chaque j, direct sur données quantisées
    static float dot_q8_0(const uint8_t* row, int n_cols, const float* x);
    static float dot_q4_0(const uint8_t* row, int n_cols, const float* x);
    static float dot_q4k (const uint8_t* row, int n_cols, const float* x);
    static float dot_q6k (const uint8_t* row, int n_cols, const float* x);
};
