/*
 * The contents of this file are subject to the terms of the Mozilla
 * Public License, v. 2.0.  If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 * Copyright 2018 - Eyal Rozenberg <E.Rozenberg@cwi.nl>
 */
#pragma once
#ifndef MONETDB_DATE_HPP_
#define MONETDB_DATE_HPP_

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <climits>

namespace monetdb {
class date_t {
public:
	using days_since_epoch_t = int;

	int dte_val; // sort of, kind of, days since Jan 1st Year 1

//	static bool parse(const char* first, const char* last, int& day, int& month, int& year);
	static constexpr date_t from_raw_days(int raw_days) { return date_t(raw_days); }

	enum : days_since_epoch_t { date_nil = INT_MIN };
		// MonetDB dates sacrifice a certain value to represent the SQL NULL value

protected:
	constexpr date_t(days_since_epoch_t val) : dte_val (val) { }

public:
	date_t() : date_t(date_nil) { }
	date_t(date_t&& d) = default;
	date_t(const date_t& d) = default;
	date_t(const char* v) : date_t(v, std::strlen(v)) { };
	date_t(const char* v, std::size_t length);
	date_t(int year, int month, int day);

public:
	bool is_nil() const { return dte_val == date_nil; }
	void add_days(int num_days) { dte_val += num_days; }
};

} // namespace monetdb
#endif
