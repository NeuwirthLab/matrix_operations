#include <getopt.h>
#include <omp.h>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include "profiler.hpp"
#include "taco.h"

namespace fs = std::filesystem;
using ValueType = double;

int main(int argc, char* argv[]) {

	fs::path input_file{};
	fs::path output_file{};
	int threads = std::thread::hardware_concurrency();

	const char* const short_opts = "i:o:t:";

	static const struct option long_opts[] = {
	    {"input", required_argument, nullptr, 'i'},
	    {"output", required_argument, nullptr, 'o'},
	    {"threads", required_argument, nullptr, 't'},
	};

	while (true) {
		const auto opt =
		    getopt_long(argc, argv, short_opts, long_opts, nullptr);
		if (opt == -1)
			break;
		switch (opt) {
			case 'i':
				input_file = optarg;
				break;
			case 'o':
				output_file = optarg;
				break;
			case 't':
				omp_set_num_threads(std::atoi(optarg));
				break;
			default:
				std::cerr << "Parameter unsupported\n";
				std::terminate();
		}
	}

	if (input_file.empty()) {
		std::cerr << "no input file given" << std::endl;
		std::terminate();
	}
	if (output_file.empty()) {
		std::cerr << "no output file name given" << std::endl;
		std::terminate();
	}

	taco::Format dm({taco::Dense, taco::Dense});
	taco::Format dv({taco::Dense});

	std::default_random_engine gen(0);
	/* std::uniform_real_distribution<ValueType> unif( */
	/*     std::numeric_limits<ValueType>::min(), */
	/*     std::numeric_limits<ValueType>::max()); */
	std::uniform_real_distribution<ValueType> unif(0, 1000);

	taco::Tensor<ValueType> A = taco::read(input_file, dm);

	taco::Tensor<ValueType> x({A.getDimension(1)}, dv);
	for (int i = 0; i < x.getDimension(0); ++i) {
		x.insert({i}, unif(gen));
	}
	x.pack();

	taco::Tensor<ValueType> z({A.getDimension(0)}, dv);
	for (int i = 0; i < z.getDimension(0); ++i) {
		z.insert({i}, unif(gen));
	}
	z.pack();

	taco::Tensor<ValueType> alpha(42.0);
	taco::Tensor<ValueType> beta(33.0);

	taco::Tensor<ValueType> y({A.getDimension(0)}, dv);

	taco::IndexVar i, j;

	y(i) = alpha() * (A(i, j) * x(j)) + beta() * z(i);

	y.compile();
	y.assemble();

	PROFILER_INIT;
	PROFILER_START("DenseMV");
	y.compute();
	PROFILER_STOP("DenseMV");
	PROFILER_CLOSE;

	taco::write(output_file, y);
	return 0;
}
