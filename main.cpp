#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#if defined(__GTEST__)
  #include <gtest/gtest.h>
#endif

#if defined(__GBENCH__)
  #include <benchmark/benchmark.h>
#endif

#define zalign(x) __attribute__((aligned(x)))
#define aligned_512_u64 __attribute__((aligned64)) uint64_t
#define load_zmm _mm512_loadu_si512
#define clmul_lo(a, b) (_mm_clmulepi64_si128((a), (b), 0))
#define clmul_hi(a, b) (_mm_clmulepi64_si128((a), (b), 17))

/* -------------------- CRC Implementations -------------------- */

__attribute__((target("sse4.2")))
inline uint32_t
crc32c_scalar(const unsigned char *data, ssize_t len, uint32_t crc)
{
    const unsigned char *p = data;
    const unsigned char *pend = p + len;

    while (p + 8 <= pend)
    {
        crc = (uint32_t) _mm_crc32_u64(crc, *((const uint64_t *) p));
        p += 8;
    }

    /* Process remaining full four bytes if any */
    if (p + 4 <= pend)
    {
        crc = _mm_crc32_u32(crc, *((const unsigned int *) p));
        p += 4;
    }

    /* go byte by byte: */
    while (p < pend)
    {
        crc = _mm_crc32_u8(crc, *p);
        p++;
    }

    return crc;
}

__attribute__((target("sse4.2,pclmul")))
inline uint32_t
crc32c_sse42_corsix(const unsigned char *buf, size_t len, uint32_t crc0)
{
    if (len >= 64) {
    /* First vector chunk. */
    __m128i x0 = _mm_loadu_si128((const __m128i*)buf), y0;
    __m128i x1 = _mm_loadu_si128((const __m128i*)(buf + 16)), y1;
    __m128i x2 = _mm_loadu_si128((const __m128i*)(buf + 32)), y2;
    __m128i x3 = _mm_loadu_si128((const __m128i*)(buf + 48)), y3;
    __m128i k;
    k = _mm_setr_epi32(0x740eef02, 0, 0x9e4addf8, 0);
    x0 = _mm_xor_si128(_mm_cvtsi32_si128(crc0), x0);
    buf += 64;
    len -= 64;
    /* Main loop. */
    while (len >= 64) {
      y0 = clmul_lo(x0, k), x0 = clmul_hi(x0, k);
      y1 = clmul_lo(x1, k), x1 = clmul_hi(x1, k);
      y2 = clmul_lo(x2, k), x2 = clmul_hi(x2, k);
      y3 = clmul_lo(x3, k), x3 = clmul_hi(x3, k);
      y0 = _mm_xor_si128(y0, _mm_loadu_si128((const __m128i*)buf)), x0 = _mm_xor_si128(x0, y0);
      y1 = _mm_xor_si128(y1, _mm_loadu_si128((const __m128i*)(buf + 16))), x1 = _mm_xor_si128(x1, y1);
      y2 = _mm_xor_si128(y2, _mm_loadu_si128((const __m128i*)(buf + 32))), x2 = _mm_xor_si128(x2, y2);
      y3 = _mm_xor_si128(y3, _mm_loadu_si128((const __m128i*)(buf + 48))), x3 = _mm_xor_si128(x3, y3);
      buf += 64;
      len -= 64;
    }
    /* Reduce x0 ... x3 to just x0. */
    k = _mm_setr_epi32(0xf20c0dfe, 0, 0x493c7d27, 0);
    y0 = clmul_lo(x0, k), x0 = clmul_hi(x0, k);
    y2 = clmul_lo(x2, k), x2 = clmul_hi(x2, k);
    y0 = _mm_xor_si128(y0, x1), x0 = _mm_xor_si128(x0, y0);
    y2 = _mm_xor_si128(y2, x3), x2 = _mm_xor_si128(x2, y2);
    k = _mm_setr_epi32(0x3da6d0cb, 0, 0xba4fc28e, 0);
    y0 = clmul_lo(x0, k), x0 = clmul_hi(x0, k);
    y0 = _mm_xor_si128(y0, x2), x0 = _mm_xor_si128(x0, y0);
    /* Reduce 128 bits to 32 bits, and multiply by x^32. */
    crc0 = _mm_crc32_u64(0, _mm_extract_epi64(x0, 0));
    crc0 = _mm_crc32_u64(crc0, _mm_extract_epi64(x0, 1));
  }
  if (len >= 16) {
    /* First vector chunk. */
    __m128i x0 = _mm_loadu_si128((const __m128i*)buf), y0;
    __m128i k;
    k = _mm_setr_epi32(0xf20c0dfe, 0, 0x493c7d27, 0);
    x0 = _mm_xor_si128(_mm_cvtsi32_si128(crc0), x0);
    buf += 16;
    len -= 16;
    /* Main loop. */
    while (len >= 16) {
      y0 = clmul_lo(x0, k), x0 = clmul_hi(x0, k);
      y0 = _mm_xor_si128(y0, _mm_loadu_si128((const __m128i*)buf)), x0 = _mm_xor_si128(x0, y0);
      buf += 16;
      len -= 16;
    }
    /* Reduce 128 bits to 32 bits, and multiply by x^32. */
    crc0 = _mm_crc32_u64(0, _mm_extract_epi64(x0, 0));
    crc0 = _mm_crc32_u64(crc0, _mm_extract_epi64(x0, 1));
  }

    return crc32c_scalar(buf, len, crc0);
}

__attribute__((target("sse4.2,pclmul")))
inline uint32_t
crc32c_sse42(const unsigned char *buf, ssize_t len, uint32_t crc)
{
    /*
     * Definitions of the bit-reflected domain constants k1,k2,k3, etc and
     * the CRC32+Barrett polynomials given at the end of the paper.
     */
    static const uint64_t zalign(16) k1k2[] = { 0x740eef02, 0x9e4addf8 };
    static const uint64_t zalign(16) k3k4[] = { 0xf20c0dfe, 0x14cd00bd6 };
    static const uint64_t zalign(16) k5k0[] = { 0xdd45aab8, 0x000000000 };
    static const uint64_t zalign(16) poly[] = { 0x105ec76f1, 0xdea713f1 };
    if (len >= 64) {
        __m128i x0, x1, x2, x3, x4, x5, x6, x7, x8, y5, y6, y7, y8;
        /*
         * There's at least one block of 64.
         */
        x1 = _mm_loadu_si128((__m128i *)(buf + 0x00));
        x2 = _mm_loadu_si128((__m128i *)(buf + 0x10));
        x3 = _mm_loadu_si128((__m128i *)(buf + 0x20));
        x4 = _mm_loadu_si128((__m128i *)(buf + 0x30));
        x1 = _mm_xor_si128(x1, _mm_cvtsi32_si128(crc));
        x0 = _mm_load_si128((__m128i *)k1k2);
        buf += 64;
        len -= 64;
        /*
         * Parallel fold blocks of 64, if any.
         */
        while (len >= 64)
        {
            x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
            x6 = _mm_clmulepi64_si128(x2, x0, 0x00);
            x7 = _mm_clmulepi64_si128(x3, x0, 0x00);
            x8 = _mm_clmulepi64_si128(x4, x0, 0x00);
            x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
            x2 = _mm_clmulepi64_si128(x2, x0, 0x11);
            x3 = _mm_clmulepi64_si128(x3, x0, 0x11);
            x4 = _mm_clmulepi64_si128(x4, x0, 0x11);
            y5 = _mm_loadu_si128((__m128i *)(buf + 0x00));
            y6 = _mm_loadu_si128((__m128i *)(buf + 0x10));
            y7 = _mm_loadu_si128((__m128i *)(buf + 0x20));
            y8 = _mm_loadu_si128((__m128i *)(buf + 0x30));
            x1 = _mm_xor_si128(x1, x5);
            x2 = _mm_xor_si128(x2, x6);
            x3 = _mm_xor_si128(x3, x7);
            x4 = _mm_xor_si128(x4, x8);
            x1 = _mm_xor_si128(x1, y5);
            x2 = _mm_xor_si128(x2, y6);
            x3 = _mm_xor_si128(x3, y7);
            x4 = _mm_xor_si128(x4, y8);
            buf += 64;
            len -= 64;
        }
        /*
         * Fold into 128-bits.
         */
        x0 = _mm_load_si128((__m128i *)k3k4);
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x1 = _mm_xor_si128(x1, x2);
        x1 = _mm_xor_si128(x1, x5);
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x1 = _mm_xor_si128(x1, x3);
        x1 = _mm_xor_si128(x1, x5);
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x1 = _mm_xor_si128(x1, x4);
        x1 = _mm_xor_si128(x1, x5);
        /*
         * Single fold blocks of 16, if any.
         */
        while (len >= 16)
        {
            x2 = _mm_loadu_si128((__m128i *)buf);
            x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
            x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
            x1 = _mm_xor_si128(x1, x2);
            x1 = _mm_xor_si128(x1, x5);
            buf += 16;
            len -= 16;
        }
        /*
         * Fold 128-bits to 64-bits.
         */
        x2 = _mm_clmulepi64_si128(x1, x0, 0x10);
        x3 = _mm_setr_epi32(~0, 0, ~0, 0);
        x1 = _mm_srli_si128(x1, 8);
        x1 = _mm_xor_si128(x1, x2);
        x0 = _mm_loadl_epi64((__m128i*)k5k0);
        x2 = _mm_srli_si128(x1, 4);
        x1 = _mm_and_si128(x1, x3);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x1 = _mm_xor_si128(x1, x2);
        /*
         * Barret reduce to 32-bits.
         */
        x0 = _mm_load_si128((__m128i*)poly);
        x2 = _mm_and_si128(x1, x3);
        x2 = _mm_clmulepi64_si128(x2, x0, 0x10);
        x2 = _mm_and_si128(x2, x3);
        x2 = _mm_clmulepi64_si128(x2, x0, 0x00);
        x1 = _mm_xor_si128(x1, x2);
        crc = _mm_extract_epi32(x1, 1);
    }
    return crc32c_scalar(buf, len, crc);
}

__attribute__((target("avx512vl,vpclmulqdq")))
inline uint32_t
crc32c_avx512(const unsigned char* data, ssize_t length, uint32_t crc)
{
	static const uint64_t zalign(64) k1k2[8] = {
		0xdcb17aa4, 0xb9e02b86, 0xdcb17aa4, 0xb9e02b86, 0xdcb17aa4,
		0xb9e02b86, 0xdcb17aa4, 0xb9e02b86};
	static const uint64_t zalign(64) k3k4[8] = {
		0x740eef02, 0x9e4addf8, 0x740eef02, 0x9e4addf8, 0x740eef02,
		0x9e4addf8, 0x740eef02, 0x9e4addf8};
	static const uint64_t zalign(64) k9k10[8] = {
		0x6992cea2, 0x0d3b6092, 0x6992cea2, 0x0d3b6092, 0x6992cea2,
		0x0d3b6092, 0x6992cea2, 0x0d3b6092};
	static const uint64_t zalign(64) k1k4[8] = {
		0x1c291d04, 0xddc0152b, 0x3da6d0cb, 0xba4fc28e, 0xf20c0dfe,
		0x493c7d27, 0x00000000, 0x00000000};

	const uint8_t *input = (const uint8_t *)data;
	if (length >= 256)
	{
		uint64_t val;
		__m512i x0, x1, x2, x3, x4, x5, x6, x7, x8, y5, y6, y7, y8;
		__m128i a1, a2;

		/*
		 * AVX-512 Optimized crc32c algorithm with mimimum of 256 bytes aligned
		 * to 32 bytes.
		 * >>> BEGIN
		 */

		/*
		* There's at least one block of 256.
		*/
		x1 = _mm512_loadu_si512((__m512i *)(input + 0x00));
		x2 = _mm512_loadu_si512((__m512i *)(input + 0x40));
		x3 = _mm512_loadu_si512((__m512i *)(input + 0x80));
		x4 = _mm512_loadu_si512((__m512i *)(input + 0xC0));
		x1 = _mm512_xor_si512(x1, _mm512_castsi128_si512(_mm_cvtsi32_si128(crc)));

		x0 = _mm512_load_si512((__m512i *)k1k2);

		input += 256;
		length -= 256;

		/*
		* Parallel fold blocks of 256, if any.
		*/
		while (length >= 256)
		{
			x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
			x6 = _mm512_clmulepi64_epi128(x2, x0, 0x00);
			x7 = _mm512_clmulepi64_epi128(x3, x0, 0x00);
			x8 = _mm512_clmulepi64_epi128(x4, x0, 0x00);

			x1 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
			x2 = _mm512_clmulepi64_epi128(x2, x0, 0x11);
			x3 = _mm512_clmulepi64_epi128(x3, x0, 0x11);
			x4 = _mm512_clmulepi64_epi128(x4, x0, 0x11);

			y5 = _mm512_loadu_si512((__m512i *)(input + 0x00));
			y6 = _mm512_loadu_si512((__m512i *)(input + 0x40));
			y7 = _mm512_loadu_si512((__m512i *)(input + 0x80));
			y8 = _mm512_loadu_si512((__m512i *)(input + 0xC0));

			x1 = _mm512_ternarylogic_epi64(x1, x5, y5, 0x96);
			x2 = _mm512_ternarylogic_epi64(x2, x6, y6, 0x96);
			x3 = _mm512_ternarylogic_epi64(x3, x7, y7, 0x96);
			x4 = _mm512_ternarylogic_epi64(x4, x8, y8, 0x96);

			input += 256;
			length -= 256;
		}

		/*
		 * Fold 256 bytes into 64 bytes.
		 */
		x0 = _mm512_load_si512((__m512i *)k9k10);
		x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
		x6 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
		x3 = _mm512_ternarylogic_epi64(x3, x5, x6, 0x96);

		x7 = _mm512_clmulepi64_epi128(x2, x0, 0x00);
		x8 = _mm512_clmulepi64_epi128(x2, x0, 0x11);
		x4 = _mm512_ternarylogic_epi64(x4, x7, x8, 0x96);

		x0 = _mm512_load_si512((__m512i *)k3k4);
		y5 = _mm512_clmulepi64_epi128(x3, x0, 0x00);
		y6 = _mm512_clmulepi64_epi128(x3, x0, 0x11);
		x1 = _mm512_ternarylogic_epi64(x4, y5, y6, 0x96);

		/*
		 * Single fold blocks of 64, if any.
		 */
		while (length >= 64)
		{
			x2 = _mm512_loadu_si512((__m512i *)input);

			x5 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
			x1 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
			x1 = _mm512_ternarylogic_epi64(x1, x2, x5, 0x96);

			input += 64;
			length -= 64;
		}

		/*
		 * Fold 512-bits to 128-bits.
		 */
                x0 = _mm512_loadu_si512((__m512i *)k1k4);
                x4 = _mm512_clmulepi64_epi128(x1, x0, 0x00);
                x3 = _mm512_clmulepi64_epi128(x1, x0, 0x11);
                x2 = _mm512_xor_si512(x3, x4);
                a1 = _mm_xor_si128(_mm512_extracti32x4_epi32(x1, 3), _mm512_extracti32x4_epi32(x2, 0));
                a1 = _mm_ternarylogic_epi64(a1, _mm512_extracti32x4_epi32(x2, 1), _mm512_extracti32x4_epi32(x2, 2), 0x96);

                /*
                 * Fold 128-bits to 32-bits.
                 */
                val = _mm_crc32_u64(0, _mm_extract_epi64(a1, 0));
                crc = (uint32_t)_mm_crc32_u64(val, _mm_extract_epi64(a1, 1));

		/*
		 * AVX-512 Optimized crc32c algorithm with mimimum of 256 bytes aligned
		 * to 32 bytes.
		 * <<< END
		 ******************************************************************/
	}

	/*
	 * Finish any remaining bytes with a simple crc32c computation
	 */
	return crc32c_sse42(input, length, crc);
}

static uint8_t randomval()
{
    return (rand() % 255);
}

/* -------------------- Benchmark wrappers (one per algorithm) -------------------- */
#if defined(__GBENCH__)
static std::vector<unsigned char> g_bench_buffer;
static const size_t g_max_default_size = 2048; // largest of default Arg values

// Each benchmark uses State.range(0) for size.
static void scalar_crc32c(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        uint32_t crc = crc32c_scalar(g_bench_buffer.data(), n, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n));
}

static void corsix_crc32c(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        uint32_t crc = crc32c_sse42_corsix(g_bench_buffer.data(), n, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n));
}

static void sse42_crc32c(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        uint32_t crc = crc32c_sse42(g_bench_buffer.data(), n, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n));
}

static void avx512_crc32c(benchmark::State& state) {
    size_t n = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        uint32_t crc = crc32c_avx512(g_bench_buffer.data(), n, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(n));
}
#endif

/* -------------------- CLI Parsing -------------------- */
struct BenchConfig {
    std::vector<std::string> algorithms;
    std::vector<size_t> sizes;
    bool run_all = false;
};

static void split_csv(const std::string &s, std::vector<std::string>& out) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
}

static std::vector<size_t> parse_sizes_csv(const std::string& s) {
    std::vector<std::string> parts;
    split_csv(s, parts);
    std::vector<size_t> sizes;
    sizes.reserve(parts.size());
    for (auto &p : parts)
        sizes.push_back(static_cast<size_t>(std::stoull(p)));
    return sizes;
}

static std::vector<std::string> default_algorithms() {
    return { "scalar", "corsix", "sse42", "avx512" };
}

static std::vector<size_t> default_sizes() {
    // Original BENCH macro sizes
    return { 64, 128, 256, 512, 1024, 2048 };
}

static void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "  --algorithm <name>          Run a single algorithm\n"
        "  --algorithms a,b,c          Run a comma-separated list of algorithms\n"
        "  --sizes n1,n2,n3            Buffer sizes (override defaults: 64,128,256,512,1024,2048)\n"
        "  --all                       Run all algorithms (default if none specified)\n"
        "  --list                      List supported algorithms\n"
        "  --help                      Show this help\n"
        "Algorithms: scalar, corsix, sse42, avx512\n";
}

static BenchConfig parse_args(int& argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--list") {
            std::cout << "Algorithms:\n";
            for (auto& a : default_algorithms())
                std::cout << "  " << a << "\n";
            std::exit(0);
        } else if (arg == "--all") {
            cfg.run_all = true;
        } else if (arg == "--algorithm") {
            if (i + 1 >= argc) throw std::runtime_error("--algorithm requires a value");
            cfg.algorithms.push_back(argv[++i]);
        } else if (arg == "--algorithms") {
            if (i + 1 >= argc) throw std::runtime_error("--algorithms requires a value");
            std::vector<std::string> list;
            split_csv(argv[++i], list);
            cfg.algorithms.insert(cfg.algorithms.end(), list.begin(), list.end());
        } else if (arg == "--sizes") {
            if (i + 1 >= argc) throw std::runtime_error("--sizes requires a value");
            cfg.sizes = parse_sizes_csv(argv[++i]);
        } else {
            // Allow other args to pass to benchmark / gtest
            continue;
        }
    }

    if (cfg.run_all || cfg.algorithms.empty())
        cfg.algorithms = default_algorithms();
    if (cfg.sizes.empty())
        cfg.sizes = default_sizes();

    std::sort(cfg.algorithms.begin(), cfg.algorithms.end());
    cfg.algorithms.erase(std::unique(cfg.algorithms.begin(), cfg.algorithms.end()), cfg.algorithms.end());
    std::sort(cfg.sizes.begin(), cfg.sizes.end());
    cfg.sizes.erase(std::unique(cfg.sizes.begin(), cfg.sizes.end()), cfg.sizes.end());
    return cfg;
}

/* -------------------- Main Functions -------------------- */
#if defined(__GBENCH__)
int main(int argc, char** argv) {
    BenchConfig cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "Argument error: " << ex.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    // Map algorithm name -> benchmark function pointer
    std::unordered_map<std::string, void(*)(benchmark::State&)> bench_map {
        { "scalar",  scalar_crc32c },
        { "corsix",  corsix_crc32c },
        { "sse42",   sse42_crc32c },
        { "avx512",  avx512_crc32c }
    };

    // Validate algorithms
    for (auto& a : cfg.algorithms) {
        if (bench_map.find(a) == bench_map.end()) {
            std::cerr << "Unknown algorithm: " << a << "\n";
            return 1;
        }
    }

    // Prepare buffer (largest requested size)
    size_t max_size = cfg.sizes.empty() ? 0 : *std::max_element(cfg.sizes.begin(), cfg.sizes.end());
    if (max_size < g_max_default_size) max_size = g_max_default_size;
    g_bench_buffer.resize(max_size);
    srand(42);
    for (size_t ii = 0; ii < max_size; ++ii) {
        g_bench_buffer[ii] = randomval();
    }

    // Dynamically register benchmarks with requested sizes.
    for (const auto& alg : cfg.algorithms) {
        for (auto sz : cfg.sizes) {
            std::string name = alg + "/size=" + std::to_string(sz);
            benchmark::RegisterBenchmark(name.c_str(), bench_map[alg])->Arg(static_cast<int>(sz));
        }
    }

    // Initialize / run
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
#elif defined(__GTEST__)
#define MAX_BUF_SIZE 4*1024

TEST(test_crc32c, sse_scalar) {
    std::vector<size_t> bufsize = std::vector<size_t>(MAX_BUF_SIZE);
    std::iota(bufsize.begin(), bufsize.end(), 1);

    /* test for all buf sizes 1 - MAX_BUF_SIZE */
    for (auto size : bufsize) {
        /* Initialize to random values */
        std::vector<unsigned char> arr(size);
        srand(42);
        for (size_t ii = 0; ii < size; ++ii) {
            arr[ii] = randomval();
        }

        /* Compute crc32c using simple scalar methods and SIMD method */
        uint32_t ssecrc = crc32c_sse42(arr.data(), size, 0xFFFFFFFF);
        uint32_t scalar_crc = crc32c_scalar(arr.data(), size, 0xFFFFFFFF);
        uint32_t corsixcrc = crc32c_sse42_corsix(arr.data(), size, 0xFFFFFFFF);

        /* ASSERT values are the same */
        ASSERT_EQ(scalar_crc, ssecrc) << "buffer size = " << arr.size();
        ASSERT_EQ(corsixcrc, ssecrc) << "buffer size = " << arr.size();
        arr.clear();
    }
}

TEST(test_crc32c, avx512_scalar) {
    std::vector<size_t> bufsize = std::vector<size_t>(MAX_BUF_SIZE);
    std::iota(bufsize.begin(), bufsize.end(), 1);

    /* test for all buf sizes 1 - MAX_BUF_SIZE */
    for (auto size : bufsize) {

        /* Initialize to random values */
        std::vector<unsigned char> arr(size);
        srand(42);
        for (size_t ii = 0; ii < size; ++ii) {
            arr[ii] = randomval();
        }

        /* Compute crc32c using simple scalar methods and SIMD method */
        uint32_t avxcrc = crc32c_avx512(arr.data(), size, 0xFFFFFFFF);
        uint32_t scalar_crc = crc32c_scalar(arr.data(), size, 0xFFFFFFFF);

        /* ASSERT values are the same */
        ASSERT_EQ(scalar_crc, avxcrc) << "buffer size = " << arr.size();
        arr.clear();
    }
}

TEST(CRC32C, Agreement) {
    const char* msg = "hello world";
    auto* p = reinterpret_cast<const unsigned char*>(msg);
    size_t n = std::strlen(msg);
    uint32_t c_scalar = crc32c_scalar(p, n, 0);
    uint32_t c_corsix = crc32c_sse42_corsix(p, n, 0);
    uint32_t c_sse42  = crc32c_sse42(p, n, 0);
    uint32_t c_avx512 = crc32c_avx512(p, n, 0);
    ASSERT_EQ(c_scalar, c_corsix);
    ASSERT_EQ(c_scalar, c_sse42);
    ASSERT_EQ(c_scalar, c_avx512);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#else
int main(int argc, char** argv) {
    print_usage(argv[0]);
    return 0;
}
#endif
