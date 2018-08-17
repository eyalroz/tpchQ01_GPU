#ifndef H_TPCH
#define H_TPCH

#include <stdint.h>
#include <string>
#include <cassert>
#include <cmath>
#include <limits>
#include "buffer.hpp"
#include "date.hpp"
#include "decimal.hpp" // Not actually used in this header, but necessary

struct SkipCol {
	SkipCol(const char* v, int64_t len) {}
};

struct Char {
	char chr_val;

	Char(const char* v, int64_t len) {
		assert(len == 1);
		assert(v[0] && !v[1]);
		chr_val = v[0];
	}
};

namespace detail {
template<typename T = int64_t>
struct MinMax {
	int64_t min = std::numeric_limits<T>::max();
	int64_t max = std::numeric_limits<T>::min();

	void operator()(const T& val) {
		if (val <= min) {
			min = val;
		}
		if (val >= max) {
			max = val;
		}
	}
};
} // namespace detail

template<typename T>
struct Column : Buffer<T> {
	size_t cardinality;

	detail::MinMax<T> minmax;

	Column(size_t init_cap)
	 : Buffer<T>(init_cap), cardinality(0) {
	}

	bool HasSpaceFor(size_t n) const {
		return cardinality + n < Buffer<T>::capacity();
	}

	void Push(const T& val) {
		if (!HasSpaceFor(1)) {
			Buffer<T>::resizeByFactor(1.5);
		}
		assert(HasSpaceFor(1));
		auto data = Buffer<T>::get();
		data[cardinality++] = val;
		minmax(val);
	}
};

// starting from 1
struct lineitem {
	Column<char> l_returnflag; // 9
	Column<char> l_linestatus; // 10
	Column<int64_t> l_quantity; // 5, DECIMAL(15,2)
	Column<int64_t> l_extendedprice; // 6, DECIMAL(15,2)
	Column<int64_t> l_discount; // 7, DECIMAL(15,2)
	Column<int64_t> l_tax; // 8, DECIMAL(15,2)
	Column<int> l_shipdate; // 11
public:
	lineitem(size_t init_cap)
	 : l_returnflag(init_cap), l_linestatus(init_cap), l_quantity(init_cap), l_extendedprice(init_cap), l_discount(init_cap), l_tax(init_cap), l_shipdate(init_cap) {
	}

	void FromFile(const std::string& file);
};

#endif
