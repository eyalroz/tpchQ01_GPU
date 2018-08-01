# TPC-H Query 01 Optimized for GPU execution

This fork of the [original repository](https://github.com/diegomestre2/tpchQ01_GPU) is intended for continued work on the GPU-side code by myself ([Eyal](https://eyalroz.github.io)). I did not have the time to code all of the implementation variants I had wanted before the [ADMS 2018 paper](https://www.researchgate.net/publication/280626545) was due, plus I was unhappy with other aspects of the code, so I've continued a bit further.

This repository should be a stand-in for the forked one, with the main differences being:

* More distinct implementations.
* Faster execution on compressed data (via caching; so the second call and onwards).
* Support for choosing one of several devices on your system.
* Slightly more structuring of the code in `main.cu`
* Better separation between general-purpose utility code and code specific to our work.
* Other stuff - [resolved](https://github.com/eyalroz/tpch_q1_on_gpu/issues?q=is%3Aissue+is%3Aclosed) and [unresolved](https://github.com/eyalroz/tpch_q1_on_gpu/issues).

#### TPC-H Query 01 execution times:

(to be filled in; for now, use the table in the published [paper[((https://www.researchgate.net/publication/280626545).)

## Prerequisites

- CUDA v9.0 or later is recommended; CUDA v8.0 will _probably_ work, but has not been tested.
- A C++14-capable compiler compatible with your version of CUDA; only GCC has been tested.
- [CMake](http://www.cmake.org/) v3.1 or later
- A Unix-like environment for the (simple) shell scripts; without it, you may need to perform a few tasks manually
- The [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers) library.

## Building the Q1 benchmark binary

Assuming you've cloned into `/path/to/tpch_q1_gpu`:

- Configure the build and generate build files using `cmake /path/to/tpchQ1`
- Build using either your default make'ing tool or with `cmake --build /path/to/tpchQ1`; this will also generate input data for Scale Factor 1 (SF 1)

## TPC-H benchmark data

The binary uses the `LINEITEM` table from the TPC-H benchmark data set. It is expected to reside in a subdirectory of where you run your binary; thus if we're in `/foo/bar` and call `bin/tpch_q1` (with scale factor 123), a `lineitem.tbl`files must reside in `foo/bar/tpch_data/123.000000`. Alternatively, if the binary has already cached the data after loading it before, `.bin` files will have been created in the same directory, e.g. `foo/bar/tpch_data/123.000000/shipdate.bin` and/or `foo/bar/tpch_data/123.000000/compressed_shipdate.bin` for speedier reading. In this case, the binary will be willing to ignore a missing tpch.

### Generating the data 

- When building, the data for TPC-H Scale Factor 1 (SF 1) is generated as one of the default targets.
- You can use the build mechanism to generate data for two more scale factors - SF 10 and SF 100 - using `make -C /path/to/tpchQ1 data_table_sf_10` or `make -C /path/to/tpchQ1 data_table_sf_100`.
- For arbitrary scale factors, invoke the `scripts/genlineitem.sh` script.



### `tpch_01` command-line options

| Switch                  | Value range                                                          | Default value | Meaning                                                                                                                                                                                                |
|-------------------------|----------------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --device                | 0 ... number of CUDA device-1                                        | 0             | Use the CUDA device with the specified index.                                                                                            | --apply-compression     | N/A                                                                  | (off)        | Use the compression schemes described on the Wiki, to reduce the amount of data for transmission over PCI/e                                                                                            |
| --print-results         | N/A                                                                  | (off)         | Print the computed aggregates to std::cout after every run. Useful for debugging result stability issues.                                                                                              |
| --use-filter-pushdown   | N/A                                                                  | (off)         | Have the CPU check the TPC-H Q1 `WHERE` clause condition, passing only that result bit vector to the GPU. It's debatable whether this is actually a "push down"  in the traditional sense of the term. |
|  --use-coprocessing     | N/A                                                                  | (off)         | Schedule some of the work to be done on the CPU and some on the GPU                                                                                                                                    |
| --hash-table-placement  | in-registers, local-mem, per-thread-shared-mem, global               |  in-registers | Memory space + granularity for the aggregation tables; see the paper itself or the code for an explanation of what this means.                                                                         |
| --sf=                   | Integral or fractional number, limited precision                     | 1             | Which scale factor subdirectory to use (to look for the data table or cached column files). For sf 123.456789, data will be expected under `tpch/123.456789`                                           |
| --streams=              | Positive integral value                                              | 4             | The number of concurrent streams to use for scheduling GPU work. You should probably not change this.                                                                                                  |
| --threads-per-block=    | Positive integral number, preferably a multiple of 32                | 256           | Number of CUDA threads per block of a scheduled kernel of the computational work.                                                                                                                      |
| --tuples-per-thread=H   | Positive integral number, preferably high                            | 1024          | The number of tuples each thread processes individually before merging results with other threads                                                                                                      |
| --tuples-per-kernel=    | Positive integral number, preferably a multiple of threads-per-block | 1024          | Every how many input tuples is a new kernel launched?                                                                                                                                                  |



## What is TPC-H Query 1?

The query text and column information is [on the Wiki](https://github.com/diegomestre2/tpchQ01_GPU/wiki/TPCH-Query-1). For further information about the benchmark it is part of, see the [Transaction Processing Council](http://www.tpc.org/)'s  [page for TPC-H](http://www.tpc.org/tpch/default.asp)

