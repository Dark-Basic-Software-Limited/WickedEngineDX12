#pragma once

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "CommonInclude.h"
#include "wiVector.h"
#include "wiSpinLock.h"

#include "Utility/offsetAllocator.hpp"

#include <mutex>
#include <atomic>
#include <memory>
#include <cassert>
#include <algorithm>
#include <deque>
#include <map>
#include <cstdio>
#include <cstdarg>
#ifdef _MSC_VER
#include <intrin.h> // GGMAX 1.46b: _InterlockedExchangePointer for the Allocation::Reset steal-guard
#endif

namespace wi::allocator
{
	// GGMAX 1.46 (reload-corruption hunt): the shared mesh suballocator was caught handing out
	// out-of-range offsets (CreateAliasingResource E_INVALIDARG -> device removed) under mass
	// mesh churn; the silent version of that failure is two live allocations overlapping = one
	// mesh's upload stomping another's bytes (records correct, pixels wrong). These guards make
	// the loud mode a graceful reject and the silent mode a named log line in alloc_tripwire.txt.
	inline bool gg_alloc_tripwire = true;              // live-range overlap tracking + logging
	inline uint32_t gg_deferred_extra_hold = 8;        // extra frames before a freed range is reusable
#ifdef _WIN32
	extern "C" __declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
	inline unsigned long gg_tripwire_tid() { return GetCurrentThreadId(); }
#else
	inline unsigned long gg_tripwire_tid() { return 0; }
#endif
	inline void gg_tripwire_log(const char* fmt, ...)
	{
		// GGMAX 1.46c: serialized + persistent-handle logging. The per-call fopen-append version
		// TORE 982 lines under concurrent multi-thread traffic in organic capture #2 — and the
		// poisoning op was almost certainly among them. Mutex + one handle + per-line sequence
		// number = lossless, strictly ordered op history. NOTE: the handle opens once in the
		// LAUNCH cwd (exe dir), so the whole session logs to ONE file — no more exe-dir/Files split.
		static std::mutex log_mutex;
		static FILE* log_file = nullptr;
		static unsigned long long log_seq = 0;
		std::scoped_lock lck(log_mutex);
		if (log_file == nullptr)
		{
#ifdef _WIN32
			log_file = _fsopen("alloc_tripwire.txt", "a", 0x40); // _SH_DENYNO: external tools can read the log while the game runs
#else
			log_file = fopen("alloc_tripwire.txt", "a");
#endif
			if (log_file == nullptr) return;
		}
		fprintf(log_file, "#%llu ", ++log_seq);
		va_list args;
		va_start(args, fmt);
		vfprintf(log_file, fmt, args);
		va_end(args);
		fflush(log_file);
	}
	// Allocation of consecutive bytes, but no freeing, instead the whole allocator can be reset
	struct LinearAllocator
	{
		uint8_t* data = nullptr;
		size_t capacity = 0;
		size_t offset = 0;

		constexpr void init(void* mem, size_t size)
		{
			data = (uint8_t*)mem;
			capacity = size;
			reset();
		}
		constexpr uint8_t* allocate(size_t size)
		{
			if (offset + size >= capacity)
				return nullptr;
			uint8_t* ptr = data + offset;
			offset += size;
			return ptr;
		}
		constexpr void free(size_t size)
		{
			size = std::min(size, offset);
			offset -= size;
		}
		constexpr void reset()
		{
			offset = 0;
		}
	};

	// Allocation and freeing of single elements of the same size
	template<typename T, size_t block_size = 256>
	struct BlockAllocator
	{
		struct Block
		{
			struct alignas(alignof(T)) RawStruct
			{
				uint8_t data[sizeof(T)];
			};
			wi::vector<RawStruct> mem;
		};
		wi::vector<Block> blocks;
		wi::vector<T*> free_list;

		template<typename... ARG>
		inline T* allocate(ARG&&... args)
		{
			if (free_list.empty())
			{
				free_list.reserve(block_size);
				Block& block = blocks.emplace_back();
				block.mem.resize(block_size);
				T* ptr = (T*)block.mem.data();
				for (size_t i = 0; i < block_size; ++i)
				{
					free_list.push_back(ptr + i);
				}
			}
			T* ptr = free_list.back();
			free_list.pop_back();
			return new (ptr) T(std::forward<ARG>(args)...);
		}
		inline void free(T* ptr)
		{
			ptr->~T();
			free_list.push_back(ptr);
		}

		inline bool is_empty() const
		{
			return (blocks.size() * block_size) == free_list.size();
		}
	};

	// Allocation and freeing of an arbitrary number of bytes, managed in pages of the same size
	//	- this is a wrapper around OffsetAllocator that adds thread safety and refcounting
	//	- also supports deferred release for suballocated GPU resources
	struct PageAllocator
	{
		uint32_t page_count = 0;
		uint32_t page_size = 0;
		struct AllocationInternal
		{
			std::atomic<int> refcount{ 0 };
			OffsetAllocator::Allocation allocation;
		};
		struct AllocatorInternal
		{
			std::mutex locker;
			OffsetAllocator::Allocator allocator;
			BlockAllocator<AllocationInternal> internal_blocks;
			bool deferred_release_enabled = false;
			uint64_t deferred_release_frame = 0;
			std::deque<std::pair<OffsetAllocator::Allocation, uint64_t>> deferred_release_queue;
			std::map<uint32_t, uint32_t> live_pages; // GGMAX 1.46: outstanding page ranges (offset -> page count) for the overlap tripwire
		};
		std::shared_ptr<AllocatorInternal> allocator; // shared ptr is used to let any allocations extend the lifeftime of the allocator

		// Returns the total size that the allocator manages:
		constexpr uint64_t total_size_in_bytes() const { return uint64_t(page_count) * uint64_t(page_size); }

		// Calculates the page count that will accomodate an allocation size request
		constexpr uint32_t page_count_from_bytes(uint64_t sizeInBytes) const { return uint32_t(align((uint64_t)sizeInBytes, (uint64_t)page_size) / (uint64_t)page_size); }

		// Initializes the allocator, only after which it can be used
		//	total_size_in_bytes	:	the allocator will manage this number of bytes
		//	page_size			:	the allocation granularity in bytes, each allocation will be aligned to this
		//	deferred_release	:	if false, allocations are freed immediately (suitable for CPU only allocations), otherwise they are freed after a number of frames passed (which should be used for GPU allocations)
		void init(uint64_t total_size_in_bytes, uint32_t page_size = 64u * 1024u, bool deferred_release = false)
		{
			this->page_size = page_size;
			this->page_count = page_count_from_bytes(total_size_in_bytes);
			allocator = std::make_shared<AllocatorInternal>();
			allocator->allocator.init(page_count, std::min(page_count, OffsetAllocator::default_maxallocations));
			allocator->deferred_release_enabled = deferred_release;
			allocator->deferred_release_frame = 0;
			allocator->deferred_release_queue.clear();
		}
		// This needs to be called every frame if deferred release is enabled:
		void update_deferred_release(uint64_t framecount, uint32_t buffercount)
		{
			if (allocator == nullptr)
				return;
			std::scoped_lock lck(allocator->locker);
			allocator->deferred_release_frame = framecount;
			// GGMAX 1.46: hold freed ranges a few extra frames — mid-frame message pumps during
			// level load can advance FRAMECOUNT without the usual fence-wait pacing, making the
			// bare buffercount window unsafe for reuse-while-GPU-still-reads.
			while (!allocator->deferred_release_queue.empty() && allocator->deferred_release_queue.front().second + buffercount + gg_deferred_extra_hold < framecount)
			{
				if (gg_alloc_tripwire)
				{
					gg_tripwire_log("R %p %u m=%u t=%lu\n", (void*)allocator.get(), (unsigned)allocator->deferred_release_queue.front().first.offset, (unsigned)allocator->deferred_release_queue.front().first.metadata, gg_tripwire_tid());
					allocator->live_pages.erase(allocator->deferred_release_queue.front().first.offset); // GGMAX 1.46
				}
				allocator->allocator.free(allocator->deferred_release_queue.front().first);
				allocator->deferred_release_queue.pop_front();
			}
		}

		struct Allocation
		{
			std::shared_ptr<AllocatorInternal> allocator; // the allocator is retained so that allocation can deallocate itself
			AllocationInternal* internal_state = nullptr; // this is pointing within the allocator which is retained by shared_ptr
			uint64_t byte_offset = ~0ull;

			Allocation()
			{
				Reset();
			}
			Allocation(const Allocation& other)
			{
				Reset();
				allocator = other.allocator;
				internal_state = other.internal_state;
				byte_offset = other.byte_offset;
				if (internal_state != nullptr)
				{
					internal_state->refcount.fetch_add(1);
				}
			}
			Allocation(Allocation&& other) noexcept
			{
				Reset();
				allocator = std::move(other.allocator);
				internal_state = std::move(other.internal_state);
				byte_offset = other.byte_offset;
				other.allocator = nullptr;
				other.internal_state = nullptr;
				other.byte_offset = ~0ull;
			}
			~Allocation()
			{
				Reset();
			}
			Allocation& operator=(const Allocation& other)
			{
				Reset();
				allocator = other.allocator;
				internal_state = other.internal_state;
				byte_offset = other.byte_offset;
				if (internal_state != nullptr)
				{
					internal_state->refcount.fetch_add(1);
				}
				return *this;
			}
			Allocation& operator=(Allocation&& other) noexcept
			{
				Reset();
				allocator = std::move(other.allocator);
				internal_state = std::move(other.internal_state);
				byte_offset = other.byte_offset;
				other.allocator = nullptr;
				other.internal_state = nullptr;
				other.byte_offset = ~0ull;
				return *this;
			}
			void Reset()
			{
				// GGMAX 1.46b: atomically claim the state pointer so a RACING second Reset on
				// the same object (e.g. cross-thread mesh delete/rebuild) can't double-decrement
				// the refcount — that underflow eventually frees someone else's node, which
				// double-inserts it into a bin and self-loops the free list (observed as the
				// same page range granted dozens of times in a row).
#ifdef _WIN32
				AllocationInternal* st = (AllocationInternal*)_InterlockedExchangePointer((void**)&internal_state, nullptr);
#else
				AllocationInternal* st = internal_state;
				internal_state = nullptr;
#endif
				std::shared_ptr<AllocatorInternal> keep = std::move(allocator); // keep alive through the free block
				if (st != nullptr && keep != nullptr && (st->refcount.fetch_sub(1) <= 1))
				{
					std::scoped_lock lck(keep->locker);
					if (keep->deferred_release_enabled)
					{
						// can only be reclaimed after buffering amount of frames passed, this is usually used for GPU resources:
						keep->deferred_release_queue.push_back(std::make_pair(st->allocation, keep->deferred_release_frame));
						// GGMAX 1.46: range stays in live_pages until the drain actually frees it
						if (gg_alloc_tripwire)
						{
							gg_tripwire_log("D %p %u m=%u t=%lu\n", (void*)keep.get(), (unsigned)st->allocation.offset, (unsigned)st->allocation.metadata, gg_tripwire_tid());
						}
					}
					else
					{
						// reclaimed immediately:
						keep->allocator.free(st->allocation);
						if (gg_alloc_tripwire)
						{
							gg_tripwire_log("F %p %u m=%u t=%lu\n", (void*)keep.get(), (unsigned)st->allocation.offset, (unsigned)st->allocation.metadata, gg_tripwire_tid());
							keep->live_pages.erase(st->allocation.offset); // GGMAX 1.46
						}
					}
					keep->internal_blocks.free(st);
				}
				allocator = {};
				byte_offset = ~0ull;
			}

			constexpr bool IsValid() const { return internal_state != nullptr; }
		};

		// Allocates a reference counted allocation, viewing at least the requested amount of bytes
		//	To check if the allocation succeeded, call IsValid() on the returned object
		inline Allocation allocate(size_t sizeInBytes)
		{
			const uint32_t pages = page_count_from_bytes(sizeInBytes);
			std::scoped_lock lck(allocator->locker);
			OffsetAllocator::Allocation offsetallocation = allocator->allocator.allocate(pages);
			Allocation alloc;
			if (offsetallocation.offset != OffsetAllocator::Allocation::NO_SPACE)
			{
				// GGMAX 1.46: an out-of-range grant means the free-list metadata is corrupt.
				// Reject it (caller falls back to a standalone buffer) instead of letting
				// CreateAliasingResource fail with E_INVALIDARG and remove the device.
				if (offsetallocation.offset + pages > page_count)
				{
					gg_tripwire_log("OOB-ALLOC offset=%u pages=%u page_count=%u (metadata corrupt) -> rejected\n",
						(unsigned)offsetallocation.offset, (unsigned)pages, (unsigned)page_count);
					allocator->allocator.free(offsetallocation);
					return alloc;
				}
				// GGMAX 1.46: overlap tripwire — a grant intersecting a LIVE range is the silent
				// double-booking that lets one mesh's upload stomp another's bytes.
				if (gg_alloc_tripwire)
				{
					gg_tripwire_log("A %p %u %u pc=%u m=%u t=%lu\n", (void*)allocator.get(), (unsigned)offsetallocation.offset, (unsigned)pages, (unsigned)page_count, (unsigned)offsetallocation.metadata, gg_tripwire_tid());
					auto next = allocator->live_pages.lower_bound(offsetallocation.offset);
					if (next != allocator->live_pages.begin())
					{
						auto prev = std::prev(next);
						if (prev->first + prev->second > offsetallocation.offset)
						{
							gg_tripwire_log("OVERLAP-ALLOC new=[%u,+%u) collides prev=[%u,+%u)\n",
								(unsigned)offsetallocation.offset, (unsigned)pages, (unsigned)prev->first, (unsigned)prev->second);
						}
					}
					if (next != allocator->live_pages.end() && next->first < offsetallocation.offset + pages)
					{
						gg_tripwire_log("OVERLAP-ALLOC new=[%u,+%u) collides next=[%u,+%u)\n",
							(unsigned)offsetallocation.offset, (unsigned)pages, (unsigned)next->first, (unsigned)next->second);
					}
					allocator->live_pages[offsetallocation.offset] = pages;
				}
				alloc.allocator = allocator;
				alloc.internal_state = allocator->internal_blocks.allocate();
				alloc.internal_state->refcount.store(1);
				alloc.internal_state->allocation = offsetallocation;
				alloc.byte_offset = offsetallocation.offset * page_size;
			}
			return alloc;
		}

		// returns true if no pages are allocated
		inline bool is_empty() const
		{
			std::scoped_lock lck(allocator->locker); // GGMAX 1.46: storageReport walks free-lists; racing a concurrent free() was a torn read
			return allocator->allocator.storageReport().totalFreeSpace == page_count;
		}
	};




	// Interface for allocating pooled shared_ptr
	struct SharedBlockAllocator
	{
		virtual void init_refcount(void* ptr) = 0;
		virtual uint32_t get_refcount(void* ptr) = 0;
		virtual uint32_t inc_refcount(void* ptr) = 0;
		virtual uint32_t dec_refcount(void* ptr) = 0;
		virtual uint32_t get_refcount_weak(void* ptr) = 0;
		virtual uint32_t inc_refcount_weak(void* ptr) = 0;
		virtual uint32_t dec_refcount_weak(void* ptr) = 0;
		virtual bool try_inc_refcount(void* ptr) = 0;
	};

	// The per-type block allocators can be indexed with bottom 8 bits of the shared_ptr's handle:
	inline SharedBlockAllocator* block_allocators[256] = {};
	inline std::atomic<uint8_t> next_allocator_id{ 0 };
	inline uint8_t register_shared_block_allocator(SharedBlockAllocator* allocator)
	{
		uint8_t id = next_allocator_id.fetch_add(1);
		assert(id < arraysize(block_allocators));
		block_allocators[id] = allocator;
		return id;
	}
	inline uint8_t get_shared_block_allocator_count() { return next_allocator_id.load(); }

	// Shared ptr using a block allocation strategy, refcounted, thread-safe, reduced size using single uint64_t handle
	//	This makes it easy to swap-out std::shared_ptr, but not feature complete, only has minimal feature set
	//	Use this if you require many object of the same type, their memory allocation will be pooled
	//	If you require just a single object, it will be better to use std::shared_ptr instead
	template<typename T>
	struct shared_ptr
	{
		uint64_t handle = 0;

		constexpr bool IsValid() const { return handle != 0; }

		constexpr T* get_ptr() const { return (T*)(handle & (~0ull << 8ull)); }
		constexpr SharedBlockAllocator* get_allocator() const { return block_allocators[handle & 0xFF]; }

		constexpr T* operator->() const { return get_ptr(); }
		constexpr operator T* () const { return get_ptr(); }
		constexpr T* get() const { return get_ptr(); }

		template<typename U>
		operator shared_ptr<U>& () const { return *(shared_ptr<U>*)this; }

		shared_ptr() = default;
		shared_ptr(const shared_ptr& other) { copy(other); }
		shared_ptr(shared_ptr&& other) noexcept { move(other); }
		~shared_ptr() noexcept { reset(); }
		shared_ptr& operator=(const shared_ptr& other) { copy(other); return *this; }
		shared_ptr& operator=(shared_ptr&& other) noexcept { move(other); return *this; }

		void reset() noexcept
		{
			if (IsValid())
			{
				get_allocator()->dec_refcount(get_ptr());
			}
			handle = 0;
		}
		void copy(const shared_ptr& other)
		{
			reset();
			handle = other.handle;
			if (IsValid())
			{
				get_allocator()->inc_refcount(get_ptr());
			}
		}
		void move(shared_ptr& other) noexcept
		{
			if (this == &other)
				return;
			reset();
			handle = other.handle;
			other.handle = 0;
		}
		uint32_t use_count() const { return IsValid() ? get_allocator()->get_refcount(get_ptr()) : 0; }
	};

	// Similar to std::weak_ptr but works with the shared block allocator, and reduced feature set
	template<typename T>
	struct weak_ptr
	{
		uint64_t handle = 0;

		constexpr bool IsValid() const { return handle != 0; }

		constexpr T* get_ptr() const { return (T*)(handle & (~0ull << 8ull)); }
		constexpr SharedBlockAllocator* get_allocator() const { return block_allocators[handle & 0xFF]; }

		template<typename U>
		operator weak_ptr<U>& () const { return *(weak_ptr<U>*)this; }

		weak_ptr() = default;
		weak_ptr(const weak_ptr& other) { copy(other); }
		weak_ptr(weak_ptr&& other) noexcept { move(other); }
		~weak_ptr() noexcept { reset(); }
		weak_ptr& operator=(const weak_ptr& other) { copy(other); return *this; }
		weak_ptr& operator=(weak_ptr&& other) noexcept { move(other); return *this; }

		weak_ptr(const shared_ptr<T>& other)
		{
			reset();
			handle = other.handle;
			if (IsValid())
			{
				get_allocator()->inc_refcount_weak(get_ptr());
			}
		}

		shared_ptr<T> lock()
		{
			if (!IsValid())
				return {};

			SharedBlockAllocator* alloc = get_allocator();
			T* ptr = get_ptr();

			// GGMAX 1.52 (upstream be8c766e): CAS-based lock. The old inc-then-undo pattern
			// double-released the strong side's collective weak claim on every failed lock of a
			// dead entry (dec_refcount's old==1 path calls dec_refcount_weak unconditionally),
			// freeing the pool block while the resourcemanager name map still referenced it —
			// the block got re-granted to the next resource (often the adjacent _normal.dds)
			// and the name->resource binding silently aliased a foreign texture.
			if (alloc->try_inc_refcount(ptr))
			{
				shared_ptr<T> ret;
				ret.handle = handle;
				return ret;
			}
			return {};
		}

		void reset() noexcept
		{
			if (IsValid())
			{
				get_allocator()->dec_refcount_weak(get_ptr());
			}
			handle = 0;
		}
		void copy(const weak_ptr& other)
		{
			reset();
			handle = other.handle;
			if (IsValid())
			{
				get_allocator()->inc_refcount_weak(get_ptr());
			}
		}
		void move(weak_ptr& other) noexcept
		{
			if (this == &other)
				return;
			reset();
			handle = other.handle;
			other.handle = 0;
		}
		uint32_t use_count() const { return IsValid() ? get_allocator()->get_refcount(get_ptr()) : 0; }
		bool expired() const noexcept
		{
			return !IsValid() || use_count() == 0;
		}
	};

	// Implementation of a thread-safe refcounted block allocator
	template<typename T, size_t block_size = 256>
	struct SharedBlockAllocatorImpl final : public SharedBlockAllocator
	{
		const uint8_t allocator_id = register_shared_block_allocator(this);

		struct alignas(std::max(size_t(256), alignof(T))) RawStruct // 256 alignment is used at least because I use bottom 8 bits of pointer as allocator id
		{
			uint8_t data[sizeof(T)];
			std::atomic<uint32_t> refcount;
			std::atomic<uint32_t> refcount_weak;
		};
		static_assert(offsetof(RawStruct, data) == 0); // we assume that data is located at 0 when casting ptr to T*, this avoids having to do a function call that would return T* like the refcounts

		struct Block
		{
			std::unique_ptr<RawStruct[]> mem;
		};
		wi::vector<Block> blocks;
		wi::vector<RawStruct*> free_list;
		//std::mutex locker;
		wi::SpinLock locker;

		template<typename... ARG>
		inline shared_ptr<T> allocate(ARG&&... args)
		{
			locker.lock();
			if (free_list.empty())
			{
				Block& block = blocks.emplace_back();
				block.mem.reset(new RawStruct[block_size]);
				RawStruct* ptr = block.mem.get();
				free_list.reserve(block_size);
				for (size_t i = 0; i < block_size; ++i)
				{
					free_list.push_back(ptr + i);
				}
			}
			RawStruct* ptr = free_list.back();
			assert((uint64_t)ptr == ((uint64_t)ptr & (~0ull << 8ull))); // The pointer lower 8 bits must be 0, it will be used as allocator index
			free_list.pop_back();
			locker.unlock();

			// Construction can be outside of lock, this structure wasn't shared yet:
			new (ptr) T(std::forward<ARG>(args)...);
			init_refcount(ptr);
			shared_ptr<T> allocation;
			allocation.handle = uint64_t(ptr) | uint64_t(allocator_id);
			return allocation;
		}

		void reclaim(void* ptr)
		{
			std::scoped_lock lck(locker);
			free_list.push_back((RawStruct*)ptr);
		}

		void init_refcount(void* ptr) override
		{
			static_cast<RawStruct*>(ptr)->refcount.store(1, std::memory_order_relaxed);
			static_cast<RawStruct*>(ptr)->refcount_weak.store(1, std::memory_order_relaxed);
		}
		uint32_t get_refcount(void* ptr) override
		{
			return static_cast<RawStruct*>(ptr)->refcount.load(std::memory_order_acquire);
		}
		uint32_t inc_refcount(void* ptr) override
		{
			return static_cast<RawStruct*>(ptr)->refcount.fetch_add(1, std::memory_order_relaxed);
		}
		uint32_t dec_refcount(void* ptr) override
		{
			uint32_t old = static_cast<RawStruct*>(ptr)->refcount.fetch_sub(1, std::memory_order_acq_rel);
			if (old == 1)
			{
				static_cast<T*>(ptr)->~T();
				dec_refcount_weak(ptr);
			}
			return old;
		}
		uint32_t get_refcount_weak(void* ptr) override
		{
			return static_cast<RawStruct*>(ptr)->refcount_weak.load(std::memory_order_acquire);
		}
		uint32_t inc_refcount_weak(void* ptr) override
		{
			return static_cast<RawStruct*>(ptr)->refcount_weak.fetch_add(1, std::memory_order_relaxed);
		}
		uint32_t dec_refcount_weak(void* ptr) override
		{
			uint32_t old = static_cast<RawStruct*>(ptr)->refcount_weak.fetch_sub(1, std::memory_order_acq_rel);
			if (old == 1)
			{
				reclaim(ptr);
			}
			return old;
		}
		bool try_inc_refcount(void* ptr) override
		{
			auto& ref = static_cast<RawStruct*>(ptr)->refcount;
			uint32_t expected = ref.load(std::memory_order_acquire);
			do {
				if (expected == 0) {
					return false;
				}
			} while (!ref.compare_exchange_weak(expected, expected + 1, std::memory_order_acq_rel, std::memory_order_acquire));
			return true;
		}
	};

	// (GGMAX 1.52 note: upstream be8c766e also introduces a SharedHeapAllocator here — not
	// ported; nothing in this vintage instantiates it, and the weak_ptr fix is complete without it.)

	// The allocators are global intentionally, this avoids runtime construction, guard check
	template<typename T, size_t block_size = 256>
	inline static SharedBlockAllocatorImpl<T, block_size>* shared_block_allocator = new SharedBlockAllocatorImpl<T, block_size>; // only destroyed after program exit, never earlier

	// Create a new shared pooled object:
	template<typename T, size_t block_size = 256, typename... ARG>
	inline shared_ptr<T> make_shared(ARG&&... args)
	{
		return shared_block_allocator<T, block_size>->allocate(std::forward<ARG>(args)...);
	}

}
