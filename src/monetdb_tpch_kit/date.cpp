/*
 * The contents of this file are subject to the terms of the Mozilla
 * Public License, v. 2.0.  If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 * Copyright 2018 - Eyal Rozenberg <E.Rozenberg@cwi.nl>
 */
#include "date.hpp"

#include <climits>
#include <cassert>

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

enum : int { int_nil = INT_MIN };

namespace monetdb {

/* phony zero values, used to get negative numbers from unsigned
 * sub-integers in rule */
#define WEEKDAY_ZERO	8
#define DAY_ZERO	32
#define OFFSET_ZERO	4096


static int LEAPDAYS[13] = {
	0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};
static int CUMDAYS[13] = {
	0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365
};
static int CUMLEAPDAYS[13] = {
	0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366
};

#define YEAR_MAX		5867411
#define YEAR_MIN		(-YEAR_MAX)
#define MONTHDAYS(m,y)	((m) != 2 ? LEAPDAYS[m] : leapyear(y) ? 29 : 28)
#define YEARDAYS(y)		(leapyear(y) ? 366 : 365)
#define DATE(d,m,y)		((m) > 0 && (m) <= 12 && (d) > 0 && (y) != 0 && (y) >= YEAR_MIN && (y) <= YEAR_MAX && (d) <= MONTHDAYS(m, y))
#define TIME(h,m,s,x)	((h) >= 0 && (h) < 24 && (m) >= 0 && (m) < 60 && (s) >= 0 && (s) < 60 && (x) >= 0 && (x) < 1000)
#define LOWER(c)		((c) >= 'A' && (c) <= 'Z' ? (c) + 'a' - 'A' : (c))


#define leapyear(y)		((y) % 4 == 0 && ((y) % 100 != 0 || (y) % 400 == 0))

static int
leapyears(int year)
{
	/* count the 4-fold years that passed since jan-1-0 */
	int y4 = year / 4;

	/* count the 100-fold years */
	int y100 = year / 100;

	/* count the 400-fold years */
	int y400 = year / 400;

	return y4 + y400 - y100 + (year >= 0);	/* may be negative */
}

static int todate(int year, int month, int day)
{
	date_t::days_since_epoch_t n = date_t::date_nil;

	if (DATE(day, month, year)) {
		if (year < 0)
			year++;				/* HACK: hide year 0 */
		n = (date_t::days_since_epoch_t) (day - 1);
		if (month > 2 && leapyear(year))
			n++;
		n += CUMDAYS[month - 1];
		/* current year does not count as leapyear */
		n += 365 * year + leapyears(year >= 0 ? year - 1 : year);
	}
	return n;
}

date_t::date_t(int year, int month, int day) : dte_val(todate(year, month, day)) { }

date_t::date_t(const char* v, std::size_t length)
{
	int day = 0, month = 0, year = 0;

    using boost::spirit::qi::int_;
    using boost::spirit::qi::_1;
    using boost::spirit::qi::phrase_parse;
    using boost::spirit::ascii::space;
    using boost::phoenix::ref;

    const char* it = v;
    const char* end = v + length;
    bool r = phrase_parse(it, end,
        ( int_[ref(year) = _1] >> '-' >> int_[ref(month) = _1] >> '-' >> int_[ref(day) = _1]),
        space);

    if (!r || it != end) // fail if we did not get a full match
        assert(false && "parsing failed");

	dte_val = todate(year, month, day);
}


void fromdate(date_t::days_since_epoch_t n, int *d, int *m, int *y)
{
	int day, month, year;

	if (n == date_t::date_nil) {
		if (d)
			*d = int_nil;
		if (m)
			*m = int_nil;
		if (y)
			*y = int_nil;
		return;
	}
	year = n / 365;
	day = (n - year * 365) - leapyears(year >= 0 ? year - 1 : year);
	if (n < 0) {
		year--;
		while (day >= 0) {
			year++;
			day -= YEARDAYS(year);
		}
		day = YEARDAYS(year) + day;
	} else {
		while (day < 0) {
			year--;
			day += YEARDAYS(year);
		}
	}
	if (d == 0 && m == 0) {
		if (y)
			*y = (year <= 0) ? year - 1 : year;	/* HACK: hide year 0 */
		return;
	}

	day++;
	if (leapyear(year)) {
		for (month = day / 31 == 0 ? 1 : day / 31; month <= 12; month++)
			if (day > CUMLEAPDAYS[month - 1] && day <= CUMLEAPDAYS[month]) {
				if (m)
					*m = month;
				if (d == 0)
					return;
				break;
			}
		day -= CUMLEAPDAYS[month - 1];
	} else {
		for (month = day / 31 == 0 ? 1 : day / 31; month <= 12; month++)
			if (day > CUMDAYS[month - 1] && day <= CUMDAYS[month]) {
				if (m)
					*m = month;
				if (d == 0)
					return;
				break;
			}
		day -= CUMDAYS[month - 1];
	}
	if (d)
		*d = day;
	if (m)
		*m = month;
	if (y)
		*y = (year <= 0) ? year - 1 : year;	/* HACK: hide year 0 */
}

} // namespace monetdb
