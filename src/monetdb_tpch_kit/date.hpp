#ifndef MONETDB_DATE_HPP_
#define MONETDB_DATE_HPP_

#include "monetdb.hpp"

#include <cstdint>
#include <cstring>
#include <cassert>

struct Date {
	static bool parse(const char* first, const char* last, int& day, int& month, int& year);

	int dte_val;

	Date(const char* v, int64_t len = -1, int plus_days = 0)
	{
		if (len < 0) {
			assert(len == -1);
			len = strlen(v);
		}
		int day = 0, month = 0, year = 0;
		if (!parse(v, v + len, day, month, year)) {
			assert(false);
		}
		dte_val = todate(day, month, year);
		dte_val += plus_days;
	}
};

#endif
