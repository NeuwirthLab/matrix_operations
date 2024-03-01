#ifndef _MATRIX_OPS_HPP_
#define _MATRIX_OPS_HPP_

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <getopt.h>

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

//template <typename Dense, typename Sparse, typename... Args>
//auto invoke_IO_fun(Dense dense, Sparse sparse, Args&&... args)
//    -> decltype(dense(args...)) {
//	return dense(args...);
//}

//template <typename Dense, typename Sparse, typename... Args>
//auto invoke_IO_fun(Dense dense, Sparse sparse, Args&&... args)
//    -> decltype(sparse(args...)) {
//	return sparse(args...);
//}

//// template <class MatrixType>
//// bool save_matrix(const MatrixType& matrix, const std::string& fname) {
//// 	return invoke_IO_fun(
//// 	    Eigen::saveMarketDense, Eigen::saveMarket, matrix, fname);
//// }

//template <class MatrixType>
//bool load_matrix(MatrixType& matrix, const std::string& fname) {
//	return invoke_IO_fun(
//	    Eigen::loadMarketDense, Eigen::loadMarket, matrix, fname);
//}

//template <class MatrixType, class MatrixMatrixOperation>
//result_t matrix_matrix_operation(const std::filesystem::path in_file,
//                                 const std::filesystem::path out_path,
//                                 MatrixMatrixOperation op) {
//	MatrixType a;
//	result_t result;

//	auto t0 = Clock::now();
//	if (!load_matrix<MatrixType>(a, in_file.string())) {
//		std::cerr << "Failed to load dense matrix\n";
//		std::terminate();
//	}
//	Duration d = Clock::now() - t0;
//	result.load_time = d.count();

//	t0 = Clock::now();
//	MatrixType b = op(a, a);
//	d = Clock::now() - t0;

//	result.compute_time = d.count();

//	t0 = Clock::now();
//	// if (!save_matrix(
//	//         b,
//	//         out_path.string() + "/" + in_file.filename().string() + ".out")) {
//	// 	std::cerr << "Failed to write dense matrix\n";
//	// 	std::terminate();
//	// }
//	d = Clock::now() - t0;

//	result.store_time = d.count();
//	result.data_size = a.size() * sizeof(ValueType);
//	//result.fp_ops = 2 * (a.rows() * a.rows() * a.cols());
//	result.path = in_file;
//	return result;
//}

} // namespace matrix_ops
#endif
