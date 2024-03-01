#include "matrix_ops.hpp"
#include <algorithm>

namespace fs = std::filesystem;
using MatrixData = std::tuple<fs::path, std::uintmax_t>;
using MatrixContainer = std::vector<MatrixData>;
using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;

using ValueType = float;
using DenseMatrix = Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>;
using SparseMatrix = Eigen::SparseMatrix<ValueType>;

static bool check = false;

template <typename Iter>
void sort_matrix_container(Iter begin, Iter end) {
	std::sort(
	    begin, end, [](const MatrixData lhs, const MatrixData rhs) -> bool {
		    return std::get<1>(lhs) < std::get<1>(rhs);
	    });
}

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
	LIKWID_MARKER_START("dense");
	DenseMatrix b = a.array().square();
	LIKWID_MARKER_STOP("dense");
	d = Clock::now() - t0;

	result.compute_time = d.count();

	if (check) {
		DenseMatrix c = b.array().sqrt();
		for (int i = 0; i < c.rows(); ++i) {
			for (int j = 0; j < c.cols(); ++j) {
				if (a(i, j) != c(i, j)) {
					std::cout << "Mismatch at position (" << i << ", " << j
					          << "): " << a(i, j) << " != " << c(i, j)
					          << std::endl;
					std::terminate();
				}
			}
		}
	}

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
	LIKWID_MARKER_START("sparse");
	SparseMatrix b = a.cwiseProduct(a);
	LIKWID_MARKER_STOP("sparse");
	d = Clock::now() - t0;

	result.compute_time = d.count();

	if (check) {
		SparseMatrix c = b.cwiseSqrt();
		Eigen::SparseMatrix<bool> checkResult = c.cwiseEqual(a);
		for (int k = 0; k < checkResult.outerSize(); ++k) {
			for (Eigen::SparseMatrix<bool>::InnerIterator it(checkResult, k);
			     it;
			     ++it) {
				if (!it.value()) {
					std::cerr << "Wrong result\n";
					std::terminate();

					break;
				}
			}
		}
	}

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

	const char* const short_opts = "i:o:r:c";

	static const struct option long_opts[] = {
	    {"input_dir", required_argument, nullptr, 'i'},
	    {"output_dir", required_argument, nullptr, 'o'},
	    {"result_dir", required_argument, nullptr, 'r'},
	    {"check", 0, nullptr, 'c'},
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
			case 'c':
				check = true;
				break;
			default:
				std::cerr << "Parameter not supported\n";
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

	sort_matrix_container(dense_matrices.begin(), dense_matrices.end());
	sort_matrix_container(sparse_matrices.begin(), sparse_matrices.end());

	matrix_ops::ResultCollector dense_results(dense_matrices.size());
	matrix_ops::ResultCollector sparse_results(sparse_matrices.size());

	LIKWID_MARKER_INIT;
	for (const auto& p : dense_matrices) {
		dense_operations(std::get<0>(p), output_dir, dense_results);
	}

	for (const auto& p : sparse_matrices) {
		sparse_operations(std::get<0>(p), output_dir, sparse_results);
	}
	LIKWID_MARKER_CLOSE;

	dense_results.print_csv(result_dir, "dense.csv");
	sparse_results.print_csv(result_dir, "sparse.csv");
	return 0;
}
