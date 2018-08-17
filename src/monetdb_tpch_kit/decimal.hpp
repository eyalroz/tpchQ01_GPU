/*
 * The contents of this file are subject to the terms of the Mozilla
 * Public License, v. 2.0.  If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 * Copyright 2018 - Eyal Rozenberg <E.Rozenberg@cwi.nl>
 */#pragma once
#pragma once
#ifndef MONETDB_DECIMAL_HPP_
#define MONETDB_DECIMAL_HPP_

#include <cstdint>
#include <cassert>

namespace monetdb {
namespace detail {
bool parse_decimal(const char* first, const char* last, int& intg, int& frac);

/* Reinvent standard math function pow() because it couldn't be a constexpr. */
template <typename T>
inline constexpr T ipow(T num, unsigned int pow)
{
	if (pow == 0) {
		return 1;
	}

	return num * ipow(num, pow - 1);
}
/*
template <typename T>
constexpr T ipow(T base, unsigned exponent, T coefficient) {
	return exponent == 0 ? coefficient :
		ipow(base * base, exponent >> 1, (exponent & 0x1) ? coefficient * base : coefficient);
}
 */
 } // namespace detail

template<std::size_t PRE = 15, std::size_t POST = 2, typename ValueT = int64_t>
struct decimal_t {
	ValueT dec_val;

	static constexpr int64_t int_digits = PRE;
	static constexpr int64_t frac_digits = POST;
	static constexpr ValueT scale = detail::ipow(10, frac_digits);
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

	decimal_t(const char* v, int64_t len) {
		const char* begin = v;
		const char* end = v + len;
		int intg = 0;
		int frac = 0;

		if (!detail::parse_decimal(begin, end, intg, frac)) {
			assert(false && "parsing failed");
		}

		dec_val = ToValue(intg, frac);

		assert(intg == GetInt(dec_val));
		// printf("org=%d new=%d frac_bits=%d\n", frac, GetFrac(dec_val), shift_amount);
		assert(frac == GetFrac(dec_val));
	}
};

using decimal64_t = decimal_t<15, 2>;

} // namespace monetdb {
#endif
