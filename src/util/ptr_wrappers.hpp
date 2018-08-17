/*
 * This file contains definitions for some general-purpose
 * code, which is inspecific to the TPC-H Q1 experiments,
 * and didn't fit elsewhere
 */
#ifndef PTR_WRAPPERS_HPP_
#define PTR_WRAPPERS_HPP_
#pragma once

#include "extra_pointer_traits.hpp"

#include <utility>
#include <functional>

template <typename T>
using plugged_unique_ptr = std::unique_ptr<T>;

template <typename T>
using plain_ptr = std::conditional_t<std::is_array<T>::value, std::decay_t<T>, std::decay_t<T>*>;


#endif // PTR_WRAPPERS_HPP_
