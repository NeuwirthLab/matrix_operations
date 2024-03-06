#include "profiler.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <getopt.h>

namespace fs = std::filesystem;
using MatrixContainer = std::vector<std::tuple<fs::path, std::uintmax_t>>;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

using ValueType = float;
using DenseMatrix = Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>;
using SparseMatrix = Eigen::SparseMatrix<ValueType>;

namespace matrix_ops {
typedef struct {
	double load_time;
	double store_time;
	double compute_time;
	std::uintmax_t data_size;
	std::uintmax_t fp_ops;
	double read_bw;
	double write_bw;
	std::filesystem::path path;
} result_t;

class ResultCollector {
	std::vector<result_t> _data;

   public:
	ResultCollector() = delete;
	ResultCollector(std::size_t _n) {
		_data.reserve(_n);
	}
	void insert_result(const result_t _r) {
		_data.push_back(_r);
	}
	auto get_size(void) const {
		return _data.size();
	}
	void print_csv(const std::filesystem::path _path,
	               const std::string fname) const {
		std::fstream s{_path.string() + "/" + fname, s.out};

		if (!s.is_open()) {
			std::cerr << "Failed to open file\n";
			std::terminate();
		}
		// Header
		s << "data_path,data_size,load_time,store_time,compute_time,fp_"
		     "ops,read_bw,write_bw\n";
		for (auto const& d : _data) {
			s << d.path.string() << "," << d.data_size << "," << d.load_time
			  << "," << d.store_time << "," << d.compute_time << "," << d.fp_ops << "," << d.read_bw << "," << d.write_bw
			  << std::endl;
		}
	}
};

} // namespace matrix_ops

void dense_operations(const fs::path in_file,
                      const fs::path out_path,
                      matrix_ops::ResultCollector& r) {
	DenseMatrix a;
	matrix_ops::result_t result;

	Eigen::setNbThreads(32);

	auto t0 = Clock::now();
	if (!Eigen::loadMarketDense(a, in_file.string())) {
		std::cerr << "Failed to load dense matrix\n";
		std::terminate();
	}
	Duration d = Clock::now() - t0;
	result.load_time = d.count();
	t0 = Clock::now();
	PROFILER_START("dense");
	DenseMatrix b = a.inverse();
	PROFILER_STOP("dense");
	d = Clock::now() - t0;

	std::cout << a << std::endl;
	std::cout << std::endl << b << std::endl;
	result.compute_time = d.count();

	t0 = Clock::now();
	if (!Eigen::saveMarketDense(
	        b,
	        out_path.string() + "/" + in_file.filename().string() + ".out")) {
		std::cerr << "Failed to write dense matrix\n";
		std::terminate();
	}
	d = Clock::now() - t0;

	result.store_time = d.count();
	result.data_size = a.size() * sizeof(ValueType);
	result.fp_ops = 2 * (a.rows() * a.rows() * a.cols());
	result.read_bw = (result.data_size * 1e-6) / result.load_time;
	result.write_bw = (result.data_size * 1e-6) / result.store_time;
	result.path = in_file;
	r.insert_result(result);
}

void sparse_operations(const fs::path in_file,
                       const fs::path out_path,
                       matrix_ops::ResultCollector& r) {
	SparseMatrix a;
	matrix_ops::result_t result;
	auto t0 = Clock::now();

	Eigen::setNbThreads(32);

	if (!Eigen::loadMarket(a, in_file.string())) {
		std::cerr << "Failed to load sparse matrix\n";
		std::terminate();
	}

	Duration d = Clock::now() - t0;
	result.load_time = d.count();

	t0 = Clock::now();
	PROFILER_START("sparse");
	SparseMatrix b = (a * a);
	PROFILER_STOP("sparse");
	d = Clock::now() - t0;

	result.compute_time = d.count();

	t0 = Clock::now();
	if (!Eigen::saveMarket(
	        b,
	        out_path.string() + "/" + in_file.filename().string() + ".out",
	        0)) {
		std::cerr << "Failed to write dense matrix\n";
		std::terminate();
	}
	d = Clock::now() - t0;

	result.store_time = d.count();
	result.data_size = a.nonZeros() * sizeof(ValueType);
	result.fp_ops = 2 * a.nonZeros();
	result.path = in_file;
	result.read_bw = (result.data_size * 1e-6) / result.load_time;
	result.write_bw = (result.data_size * 1e-6) / result.store_time;
	r.insert_result(result);
}

int main(int argc, char* argv[]) {

	fs::path input_dir = fs::current_path();
	fs::path output_dir = fs::current_path();
	fs::path result_dir = fs::current_path();

	const char* const short_opts = "i:o:r:";

	static const struct option long_opts[] = {
	    {"input_dir", required_argument, nullptr, 'i'},
	    {"output_dir", required_argument, nullptr, 'o'},
	    {"result_dir", required_argument, nullptr, 'r'},
	};

	while (true) {
		const auto opt =
		    getopt_long(argc, argv, short_opts, long_opts, nullptr);
		if (opt == -1)
			break;
		switch (opt) {
			case 'i':
				input_dir = optarg;
				break;
			case 'o':
				output_dir = optarg;
				break;
			case 'r':
				result_dir = optarg;
				break;
			default:
				std::cerr << "Parameter unsupported\n";
				std::terminate();
		}
	}

	MatrixContainer dense_matrices;
	MatrixContainer sparse_matrices;

	dense_matrices.reserve(100);
	sparse_matrices.reserve(100);

	for (auto const& dir_entry : fs::directory_iterator{input_dir}) {
		const std::string file_name = dir_entry.path().filename();
		const auto file_size = dir_entry.file_size();

		if (file_name.find("dense") != file_name.npos) {
			dense_matrices.push_back({dir_entry.path(), file_size});
		}
		else if (file_name.find("sparse") != file_name.npos) {
			sparse_matrices.push_back({dir_entry.path(), file_size});
		}
	}

	matrix_ops::ResultCollector dense_results(dense_matrices.size());
	matrix_ops::ResultCollector sparse_results(sparse_matrices.size());
	PROFILER_INIT;
	for (const auto& p : dense_matrices) {
		dense_operations(std::get<0>(p), output_dir, dense_results);
	}

	for (const auto& p : sparse_matrices) {
		sparse_operations(std::get<0>(p), output_dir, sparse_results);
	}
	PROFILER_CLOSE;
	dense_results.print_csv(result_dir, "dense.csv");
	sparse_results.print_csv(result_dir, "sparse.csv");
	return 0;
}
