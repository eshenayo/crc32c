#include <immintrin.h>
#include <gtest/gtest.h>
#include <numeric>

#define zalign(x) __attribute__((aligned(x)))
#define aligned_512_u64 __attribute__((aligned64)) uint64_t
#define load_zmm _mm512_loadu_si512
#define clmul_lo(a, b) (_mm_clmulepi64_si128((a), (b), 0))
#define clmul_hi(a, b) (_mm_clmulepi64_si128((a), (b), 17))

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

#ifdef __GTEST__
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
#endif

#ifdef __GBENCH__
#include <benchmark/benchmark.h>

static void scalar_crc32c(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<unsigned char> arr(size);
    srand(42);
    for (size_t ii = 0; ii < size; ++ii) {
        arr[ii] = randomval();
    }
    for (auto _ : state) {
        auto crc = crc32c_scalar(arr.data(), size, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
}

static void sse42_crc32c(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<unsigned char> arr(size);
    srand(42);
    for (size_t ii = 0; ii < size; ++ii) {
        arr[ii] = randomval();
    }
    for (auto _ : state) {
        auto crc = crc32c_sse42(arr.data(), size, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
}

static void avx512_crc32c(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<unsigned char> arr(size);
    srand(42);
    for (size_t ii = 0; ii < size; ++ii) {
        arr[ii] = randomval();
    }
    for (auto _ : state) {
        auto crc = crc32c_avx512(arr.data(), size, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
}

static void corsix_crc32c(benchmark::State& state) {
    size_t size = state.range(0);
    std::vector<unsigned char> arr(size);
    srand(42);
    for (size_t ii = 0; ii < size; ++ii) {
        arr[ii] = randomval();
    }
    for (auto _ : state) {
        auto crc = crc32c_sse42_corsix(arr.data(), size, 0xFFFFFFFF);
        benchmark::DoNotOptimize(crc);
    }
}
// Register the function as a benchmark
 #define BENCH(func) \
     BENCHMARK(func)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048);

     BENCH(scalar_crc32c)
     BENCH(sse42_crc32c)
     BENCH(avx512_crc32c)
     BENCH(corsix_crc32c)
#endif // __BENCH__
