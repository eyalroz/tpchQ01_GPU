#ifndef H_ALLOCATOR
#define H_ALLOCATOR

#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include <cstdlib>
#include <system_error>

template<typename ItemT>
class Buffer {
public:
	ItemT* m_ptr;
	size_t m_capacity;

private:
	void allocate(size_t capacity) {
		auto size_to_allocate = capacity * sizeof(ItemT);
#if __cplusplus >= 201701L
	 	m_ptr = std::aligned_alloc(size_to_allocate);
#elif _ISOC11_SOURCE
	 	m_ptr = static_cast<ItemT*>(aligned_alloc(getpagesize(), size_to_allocate));
#elif _POSIX_C_SOURCE >= 200112L
	 	auto result = posix_memalign(&static_cast<void*>(m_ptr), getpagesize(), size_to_allocate);
	 	if (result != 0) {
	 		 throw std::system_error(errno, std::generic_category(), "Failed making an aligned allocation using posix_memalign");
	 	}
#else
#error "We're too lazy to implement aligned allocation. Your compiler must support at least one of: C++17, C11, or POSIX 2001.12"
#endif
	 	assert(m_ptr);
	}

	void free() {
#if __cplusplus >= 201701L
	 	std::free(m_ptr);
#else
	 	::free(m_ptr);
#endif
	}

public:
	ItemT val;

	Buffer(size_t init_cap) : m_capacity(init_cap) { allocate(init_cap); }

	//Buffer(const Buffer& b) = delete;
	//Buffer(Buffer&& o) = delete;

	//Buffer& operator=(const Buffer&) & = delete;
	//Buffer& operator=(Buffer&&) & = delete;

	~Buffer() {
		free();
	}

	void resize(size_t new_cap) {
		m_capacity = new_cap;
		free();
		allocate(new_cap);
	}

	void resizeByFactor(float factor) {
		resize((float)m_capacity * factor);
	}

	void set_zero() {
		memset(m_ptr, 0, m_capacity * sizeof(ItemT));
	}

	size_t size() const noexcept {
		return m_capacity;
	}

	size_t capacity() const noexcept {
		return size();
	}

	ItemT* get() const noexcept {
		return (ItemT*)__builtin_assume_aligned(m_ptr, 64);
	}
};

#endif
