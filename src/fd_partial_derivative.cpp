#include "fd_partial_derivative.h"

//   dir  index indicating direction: 0-->x, 1-->y, 2-->z
//   D  m by nx*ny*nz sparse partial derivative matrix, where:
//     m = (nx-1)*ny*nz  if dir = 0
//     m = nx*(ny-1)*nz  if dir = 1
//     m = nx*ny*(nz-1)  otherwise (if dir = 2)
//
void fd_partial_derivative(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const int dir,
  Eigen::SparseMatrix<double> & D)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  ////////////////////////////////////////////////////////////////////////////
  Eigen::RowVector3i dim;
  dim << nx, ny, nz;
  dim(dir) -= 1;

  D.resize(dim.prod(), nx*ny*nz);
  std::vector<Eigen::Triplet<double>> derivative_list(dim.prod() * 2);

  auto get_col_index = [=](int i, int j, int k) {
    return i + j * nx + k * nx * ny;
  };
  auto get_row_index = [=](int i, int j, int k) {
    return i + j * dim(0) + k * dim(0) * dim(1);
  };

  for (int i_index = 0; i_index < dim(0); i_index++) {
    for (int j_index = 0; j_index < dim(1); j_index++) {
      for (int k_index = 0; k_index < dim(2); k_index++) {
        Eigen::RowVector3i end_index;
        end_index << i_index, j_index, k_index;
        end_index(dir) += 1;

        derivative_list.push_back(
          Eigen::Triplet<double>(
            get_row_index(i_index, j_index, k_index),
            get_col_index(i_index, j_index, k_index),
            -1 / h
          )
        );
        // Only if I know how to unpack Eigen vector like tuples...
        derivative_list.push_back(
          Eigen::Triplet<double>(
            get_row_index(i_index, j_index, k_index),
            get_col_index(end_index(0), end_index(1), end_index(2)),
            1 / h
          )
        );
      }
    }
  }

  D.setFromTriplets(derivative_list.begin(), derivative_list.end());
}
