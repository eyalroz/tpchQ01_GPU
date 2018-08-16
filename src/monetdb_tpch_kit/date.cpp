#include "date.hpp"

#include <cassert>

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

bool
Date::parse(const char* first, const char* last, int& day, int& month, int& year)
{
    using boost::spirit::qi::int_;
    using boost::spirit::qi::_1;
    using boost::spirit::qi::phrase_parse;
    using boost::spirit::ascii::space;
    using boost::phoenix::ref;

    bool r = phrase_parse(first, last,
        ( int_[ref(year) = _1] >> '-' >> int_[ref(month) = _1] >> '-' >> int_[ref(day) = _1]),
        space);

    if (!r || first != last) // fail if we did not get a full match
        return false;

    return r;
}

