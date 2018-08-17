/*
 * The contents of this file are subject to the terms of the Mozilla
 * Public License, v. 2.0.  If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Copyright 1997 - July 2008 CWI, August 2008 - 2016 MonetDB B.V.
 * Copyright 2018 - Eyal Rozenberg <E.Rozenberg@cwi.nl>
 */
#include "decimal.hpp"

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

namespace monetdb {
namespace detail {

bool
parse_decimal(const char* first, const char* last, int& intg, int& frac)
{
    using boost::spirit::qi::int_;
    using boost::spirit::qi::_1;
    using boost::spirit::qi::phrase_parse;
    using boost::spirit::ascii::space;
    using boost::phoenix::ref;

    bool r = phrase_parse(first, last,
        ( int_[ref(intg) = _1] >> '.' >> int_[ref(frac) = _1] | int_[ref(intg) = _1] ),
        space);

    if (!r || first != last) // fail if we did not get a full match
        return false;
    return r;
}

} // namespace detail
} // namespace monetdb
