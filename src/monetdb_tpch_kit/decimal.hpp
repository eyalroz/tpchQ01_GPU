#pragma once
#ifndef MONETDB_DECIMAL_HPP_
#define MONETDB_DECIMAL_HPP_

#include <cstdint>
#include <cassert>

struct BaseDecimal {
	static bool parse(const char* first, const char* last, int& intg, int& frac);
};

/* Reinvent standard math function pow() because it couldn't be a constexpr. */
template <typename T>
inline constexpr T ipow(T num, unsigned int pow)
{
	if (pow == 0) {
		return 1;
	}

	return num * ipow(num, pow - 1);
}

template<std::size_t PRE = 15, std::size_t POST = 2, typename ValueT = int64_t>
struct Decimal : BaseDecimal {
	ValueT dec_val;

	static constexpr int64_t int_digits = PRE;
	static constexpr int64_t frac_digits = POST;
	static constexpr ValueT scale = ipow(10, frac_digits);
	static constexpr int64_t highest_num = scale - 1;

	static constexpr ValueT ToValue(int64_t i, int64_t f) {
		return scale * (ValueT)i + f;
	}

	static constexpr int64_t GetFrac(ValueT v) {
		return v % scale;
	}

	static constexpr int64_t GetInt(ValueT v) {
		return v / scale;
	}

	static constexpr ValueT Mul(ValueT a, ValueT b) {
		return a * b; // scale will be wrong
	}

	Decimal(const char* v, int64_t len) {
		const char* begin = v;
		const char* end = v + len;
		int intg = 0;
		int frac = 0;

		if (!parse(begin, end, intg, frac)) {
			assert(false && "parsing failed");
		}

		dec_val = ToValue(intg, frac);

		assert(intg == GetInt(dec_val));
		// printf("org=%d new=%d frac_bits=%d\n", frac, GetFrac(dec_val), shift_amount);
		assert(frac == GetFrac(dec_val));
	}
};

using Decimal64 = Decimal<15, 2>;

#endif
