#ifndef MONETDB_DATE_HPP_
#define MONETDB_DATE_HPP_

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <climits>

class Date {
public:

	int dte_val; // sort of, kind of, days since Jan 1st Year 1

//	static bool parse(const char* first, const char* last, int& day, int& month, int& year);
	static constexpr Date from_raw_days(int raw_days) { return Date(raw_days); }

	enum { date_nil	= INT_MIN };
		// MonetDB dates sacrifice a certain value to represent the SQL NULL value

protected:
	constexpr Date(int val) : dte_val (val) { }

public:
	Date() : Date(date_nil) { }
	Date(Date&& d) = default;
	Date(const Date& d) = default;
	Date(const char* v) : Date(v, std::strlen(v)) { };
	Date(const char* v, std::size_t length);
	Date(int year, int month, int day);

public:
	bool is_nil() const { return dte_val == date_nil; }
	void add_days(int num_days) { dte_val += num_days; }
};

#endif
