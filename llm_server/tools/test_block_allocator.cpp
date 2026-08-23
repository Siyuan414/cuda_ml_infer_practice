/**
 * test_block_allocator.cpp — S2.6 unit tests. No GPU required.
 *
 * Build:  g++ -std=c++17 -I src tools/test_block_allocator.cpp -o build/test_alloc
 * Run:    ./build/test_alloc
 */

#include "block_allocator.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;

#define CHECK(cond, msg)                                                   \
    do { if (!(cond)) { printf("  FAIL %s  (%s:%d)\n", msg, __FILE__,      \
                               __LINE__); ++failures; } } while (0)
#define CHECK_EQ(a, b, msg)                                                \
    do { auto _a = (a); auto _b = (b);                                     \
         if (_a != _b) { printf("  FAIL %s: got %lld want %lld (%s:%d)\n", \
                                msg, (long long)_a, (long long)_b,         \
                                __FILE__, __LINE__); ++failures; } } while (0)

int main() {
    // ── blocks_for: ceiling division ─────────────────────────────────────────
    {
        printf("blocks_for\n");
        BlockAllocator a;
        a.configure({16, 100, 0});
        CHECK_EQ(a.blocks_for(0),  0, "0 tokens");
        CHECK_EQ(a.blocks_for(1),  1, "1 token needs a whole block");
        CHECK_EQ(a.blocks_for(16), 1, "exactly one block");
        CHECK_EQ(a.blocks_for(17), 2, "one over");
        CHECK_EQ(a.blocks_for(35), 3, "35 -> 3 blocks (48 slots)");
    }

    // ── Growth: a block is added ONLY at multiples of block_size ─────────────
    {
        printf("append_token boundaries\n");
        BlockAllocator a;
        a.configure({16, 100, 0});
        a.allocate(1, 1);                          // 1 token -> 1 block
        CHECK_EQ(a.table(1).size(), 1u, "prompt of 1");

        // grow 1 -> 100, checking the block count after every append
        for (int len = 1; len < 100; ++len) {
            CHECK(a.append_token(1, len), "append should succeed");
            const size_t want = (size_t)a.blocks_for(len + 1);
            CHECK_EQ(a.table(1).size(), want, "block count tracks length");
        }
    }

    // ── allocate() and append_token() must agree ────────────────────────────
    {
        printf("allocate/append agreement\n");
        BlockAllocator a;
        a.configure({16, 100, 0});
        a.allocate(1, 35);                          // 3 blocks = 48 slots
        CHECK_EQ(a.table(1).size(), 3u, "35 tokens -> 3 blocks");
        const int before = a.num_free();
        for (int len = 35; len < 48; ++len) a.append_token(1, len);
        CHECK_EQ(a.num_free(), before, "no new block until 48 slots are used");
        a.append_token(1, 48);                      // now full -> new block
        CHECK_EQ(a.table(1).size(), 4u, "block added at the boundary");
    }

    // ── Free list accounting ────────────────────────────────────────────────
    {
        printf("free list\n");
        BlockAllocator a;
        a.configure({16, 10, 0});
        CHECK_EQ(a.num_free(), 10, "starts full");
        a.allocate(1, 32);                          // 2
        a.allocate(2, 48);                          // 3
        CHECK_EQ(a.num_free(), 5, "5 handed out");
        a.release(1);
        CHECK_EQ(a.num_free(), 7, "released 2");
        a.release(2);
        CHECK_EQ(a.num_free(), 10, "all back");
        CHECK(!a.has(1) && !a.has(2), "tables erased");
    }

    // ── No external fragmentation: interleaved alloc/free stays usable ──────
    {
        printf("no external fragmentation\n");
        BlockAllocator a;
        a.configure({16, 8, 0});
        for (uint64_t i = 1; i <= 8; ++i) a.allocate(i, 16);   // 1 block each
        CHECK_EQ(a.num_free(), 0, "pool exhausted");
        a.release(2); a.release(5); a.release(7);              // scattered frees
        CHECK_EQ(a.num_free(), 3, "3 free, non-adjacent");
        // A 3-block request must still succeed — uniform blocks mean any free
        // block satisfies any need, unlike variable-size allocation.
        CHECK(a.allocate(99, 48), "3 scattered blocks are fully usable");
    }

    // ── Watermark: admission leaves room for running sequences ──────────────
    {
        printf("watermark\n");
        BlockAllocator a;
        a.configure({16, 10, 3});                   // reserve 3
        CHECK(a.can_admit(16 * 7), "7 blocks + 3 reserve == 10, fits");
        CHECK(!a.can_admit(16 * 8), "8 blocks would eat the reserve");
        a.allocate(1, 16 * 7);
        CHECK_EQ(a.num_free(), 3, "reserve intact");
        CHECK(!a.can_admit(16), "nothing admissible now");
        // ...but a RUNNING sequence can still grow into the reserve.
        CHECK(a.append_token(1, 112), "running sequence may use the reserve");
    }

    // ── Exhaustion returns false rather than throwing ────────────────────────
    {
        printf("exhaustion\n");
        BlockAllocator a;
        a.configure({16, 2, 0});
        a.allocate(1, 32);                          // takes both
        CHECK_EQ(a.num_free(), 0, "empty");
        CHECK(!a.append_token(1, 32), "append fails -> caller must preempt");
        CHECK(!a.allocate(2, 16), "allocate fails");
    }

    // ── locate(): the translation the kernel performs ───────────────────────
    {
        printf("locate\n");
        BlockAllocator a;
        a.configure({16, 100, 0});
        a.allocate(1, 35);
        const auto& t = a.table(1);
        CHECK_EQ(a.locate(1, 0).first,  t[0], "token 0 -> block 0");
        CHECK_EQ(a.locate(1, 0).second, 0,    "offset 0");
        CHECK_EQ(a.locate(1, 15).first, t[0], "token 15 still block 0");
        CHECK_EQ(a.locate(1, 15).second, 15,  "offset 15");
        CHECK_EQ(a.locate(1, 16).first, t[1], "token 16 -> block 1");
        CHECK_EQ(a.locate(1, 16).second, 0,   "offset wraps");
        CHECK_EQ(a.locate(1, 34).first, t[2], "token 34 -> block 2");
    }

    // ── flatten(): what gets uploaded to the GPU ────────────────────────────
    {
        printf("flatten\n");
        BlockAllocator a;
        a.configure({16, 100, 0});
        a.allocate(1, 35);      // 3 blocks
        a.allocate(2, 20);      // 2 blocks
        const int max_blocks = 4;
        auto flat = a.flatten({1, 2, 999}, max_blocks);   // 999 does not exist
        CHECK_EQ(flat.size(), (size_t)3 * max_blocks, "shape [3, 4]");
        CHECK_EQ(flat[0 * max_blocks + 0], a.table(1)[0], "seq 1 block 0");
        CHECK_EQ(flat[0 * max_blocks + 3], -1, "seq 1 padded");
        CHECK_EQ(flat[1 * max_blocks + 1], a.table(2)[1], "seq 2 block 1");
        CHECK_EQ(flat[1 * max_blocks + 2], -1, "seq 2 padded");
        CHECK_EQ(flat[2 * max_blocks + 0], -1, "missing seq is all -1");
    }

    // ── Utilization: the number that justifies Stage 2B ─────────────────────
    {
        printf("utilization\n");
        BlockAllocator a;
        a.configure({16, 1000, 0});
        std::vector<int> lens;
        for (uint64_t i = 1; i <= 24; ++i) { a.allocate(i, 55); lens.push_back(55); }
        const double u = a.utilization(lens);
        // 55 tokens -> 4 blocks -> 64 slots, so 55/64 = 86%
        CHECK(u > 0.85 && u < 0.87, "paged utilization ~86%");
        printf("  paged   : %.1f%%   (55-token requests, 16-token blocks)\n", 100 * u);
        printf("  2A slots: %.1f%%   (55 tokens in a 512 window)\n", 100.0 * 55 / 512);
    }

    printf("\n%s\n", failures ? "FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
