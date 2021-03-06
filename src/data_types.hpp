#pragma once

#include "util/preprocessor_shorthands.hpp"

#include <cstdint>
#include <cstddef>
#include <exception>

using std::size_t;

using ship_date_t            = int32_t;
using discount_t             = int64_t;
using extended_price_t       = int64_t;
using tax_t                  = int64_t;
using quantity_t             = int64_t;
using return_flag_t          = char; // Perhaps uint8_t? see tpch_kit.hpp
using line_status_t          = char; // Perhaps uint8_t? see tpch_kit.hpp

using sum_quantity_t         = uint64_t;
using sum_base_price_t       = uint64_t;
using sum_discounted_price_t = uint64_t;
using sum_charge_t           = uint64_t;
using sum_discount_t         = uint64_t;

using cardinality_t          = uint32_t; // Limiting ourselves to SF 500 here

namespace compressed {

using ship_date_t            = uint16_t;
using discount_t             = uint8_t;
using extended_price_t       = uint32_t;
using tax_t                  = uint8_t;
using quantity_t             = uint8_t;
using return_flag_t          = uint8_t; // Don't use this!
using line_status_t          = uint8_t; // Don't use this!
} // namespace compressed

using bit_container_t             = uint32_t;
// static_assert(std::is_same<bit_container_t,uint32_t>{}, "Expecting the bit container to hold 32 bits");

// Used to distinguish between uncompressed and compressed types,
// which is why this anonymous enum has been moved here
enum : bool { is_compressed = true, is_not_compressed = false};


/**
 * Applies a DICT(1 bit) encoding scheme to line status values
 */
__fhd__ uint8_t encode_line_status(line_status_t line_status)
{
#ifdef NDEBUG
	return (line_status == 'F') ? 0 : 1;
#else
	switch(line_status) {
	case 'F': return 0;
	case 'O': return 1;
	default:  return 0xFF;
	}
#endif
}

/**
 * Applies a DICT(2 bit) encoding scheme to return flag values
 */
__fhd__ uint8_t encode_return_flag(return_flag_t return_flag)
{
#ifdef NDEBUG
    return ((return_flag == 'R') << 1) + (return_flag == 'N');
#else
	switch(return_flag) {
	case 'A' : return 0b00;
	case 'N' : return 0b01;
	case 'R' : return 0b10;
	default  : return 0xFF;
	}
#endif
}

/**
 * Decodes using a DICT(1 bit) encoding scheme;
 * arbitrary output for inputs about 0x1
 */
__fhd__ line_status_t decode_line_status(uint8_t encoded_status)
{
#ifdef NDEBUG
	return encoded_status == 0 ? 'F' : 'O';
#else
	switch(encoded_status) {
	case 0b0: return 'F';
	case 0b1: return 'O';
	default:  return '-';
	}
#endif
}

/**
 * Decodes using a DICT(2 bit) encoding scheme; but -
 * the dictionary actually only has 3 values;
 * arbitrary output for inputs of 0x11 and above
 */
__fhd__ return_flag_t decode_return_flag(char encoded_flag)
{
#ifdef NDEBUG
    return (encoded_flag == 0) ? 'A' : ((encoded_flag == 1) ? 'N' : 'R');
#else
	switch(encoded_flag) {
	case 0b00: return 'A';
	case 0b01: return 'N';
	case 0b10: return 'R';
	default:   return '-';
	}
#endif
}


#undef __fhd__
#undef __fd__
