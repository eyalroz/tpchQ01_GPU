/** @file extra_pointer_traits.hpp
 *
 * Additional pointer traits for compound pointers over those in std::pointer_traits
 */

#pragma once
#ifndef POINTER_TRAITS_HPP_
#define POINTER_TRAITS_HPP_

#include <memory>

template <typename Ptr>
struct extra_pointer_traits;

template <typename T>
struct extra_pointer_traits<T*>
{
    using ptr_type = T*;
    using element_type = T;

    static ptr_type make(size_t count) { return new T[count]; }
    static ptr_type make()             { return new T;        }
};

template <typename T>
struct extra_pointer_traits<T[]>
{
    using ptr_type = T[];
    using element_type = T;

    static T* make(size_t count) { return new T[count]; }
    static T* make()             { return new T;        }
};

template <typename T, class Deleter>
struct extra_pointer_traits<std::unique_ptr<T[], Deleter>>
{
    using ptr_type = std::unique_ptr<T[], Deleter>;
    using element_type = T;

    static T* release(ptr_type& ptr) { return ptr.release(); }
    static ptr_type make(size_t count)
    {
        return std::make_unique<T[]>(count);
    }
};

template <typename T, class Deleter>
struct extra_pointer_traits<std::unique_ptr<T, Deleter>>
{
    using ptr_type = std::unique_ptr<T, Deleter>;
    using element_type = T;

    static T* release(ptr_type& ptr) { return ptr.release(); }
    static ptr_type make()
    {
        return std::make_unique<T>();
    }
};


template <typename T>
struct extra_pointer_traits<std::shared_ptr<T>>
{
    using ptr_type = std::shared_ptr<T>;
    using element_type = typename std::pointer_traits<ptr_type>::element_type;

    // No release()
    static ptr_type make(size_t count)
    {
        return std::make_shared<element_type[]>(count);
    }
    static ptr_type make()
    {
        return std::make_shared<element_type>();
    }
};

#endif // POINTER_TRAITS_HPP_
