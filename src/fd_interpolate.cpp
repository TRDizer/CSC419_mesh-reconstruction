#include "fd_interpolate.h"
#include <iostream>

#define INTERPOLATE(x)    (1 - x)
// Construct a matrix of trilinear interpolation weights for a
// finite-difference grid at a given set of points
//
// Inputs:
//   nx  number of grid steps along the x-direction
//   ny  number of grid steps along the y-direction
//   nz  number of grid steps along the z-direction
//   h  grid step size
//   corner  list of bottom-left-front corner position of grid
//   P  n by 3 list of query point locations
// Outputs:
//   W  n by (nx*ny*nz) sparse weights matrix
//

void fd_interpolate(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const Eigen::RowVector3d & corner,
  const Eigen::MatrixXd & P,
  Eigen::SparseMatrix<double> & W)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  ////////////////////////////////////////////////////////////////////////////
  Eigen::RowVector3d normalized_p, ratio;
  Eigen::RowVector3i cube_coordinate;
  // Indexing: g(i+j*n_x+k*n_y*n_x)
  auto get_index = [=](int i, int j, int k) {
    return i + j * nx + k * nx * ny;
  };

  W.resize(P.rows(), nx*ny*nz);

  std::vector<Eigen::Triplet<double>> weight_list(8 * P.rows());
  // W.reserve(Eigen::RowVector3d::Constant(P.rows(), 8));

  // Speed up filling speed https://stackoverflow.com/questions/17877243/filling-sparse-matrix-in-eigen-is-very-slow
  for (int row_index = 0; row_index < P.rows(); row_index++) {
    normalized_p = ((P.row(row_index) - corner) / h);
    cube_coordinate = ((P.row(row_index) - corner) / h).cast <int> ();
    ratio = normalized_p - cube_coordinate.cast <double> ();
    // std::cout << ratio << std::endl;

    // Cannot figure out the cause of assertion failure with coeffRef or insert, opt for triplet method from the same Eigen tutorial
    // W.coeffRef(row_index, get_index(cube_coordinate(0), cube_coordinate(1), cube_coordinate(2))) = INTERPOLATE(ratio(0)) * INTERPOLATE(ratio(1)) * INTERPOLATE(ratio(2));
    // W.coeffRef(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1), cube_coordinate(2))) = ratio(0) * INTERPOLATE(ratio(1)) * INTERPOLATE(ratio(2));

    // W.coeffRef(row_index, get_index(cube_coordinate(0), cube_coordinate(1) + 1, cube_coordinate(2))) = INTERPOLATE(ratio(0)) * ratio(1) * INTERPOLATE(ratio(2));
    // W.coeffRef(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1) + 1, cube_coordinate(2))) = ratio(0) * ratio(1) * INTERPOLATE(ratio(2));

    // W.coeffRef(row_index, get_index(cube_coordinate(0), cube_coordinate(1), cube_coordinate(2) + 1)) = INTERPOLATE(ratio(0)) * INTERPOLATE(ratio(1)) * ratio(2);
    // W.coeffRef(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1), cube_coordinate(2) + 1)) = ratio(0) * INTERPOLATE(ratio(1)) * ratio(2);

    // W.coeffRef(row_index, get_index(cube_coordinate(0), cube_coordinate(1) + 1, cube_coordinate(2)) + 1) = INTERPOLATE(ratio(0)) * ratio(1) * ratio(2);
    // W.coeffRef(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1) + 1, cube_coordinate(2)) + 1) = ratio.prod();

    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0), cube_coordinate(1), cube_coordinate(2)),             INTERPOLATE(ratio(0)) * INTERPOLATE(ratio(1)) * INTERPOLATE(ratio(2))));
    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1), cube_coordinate(2)),         ratio(0) * INTERPOLATE(ratio(1)) * INTERPOLATE(ratio(2))));

    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0), cube_coordinate(1) + 1, cube_coordinate(2)),         INTERPOLATE(ratio(0)) * ratio(1) * INTERPOLATE(ratio(2))));
    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1) + 1, cube_coordinate(2)),     ratio(0) * ratio(1) * INTERPOLATE(ratio(2))));

    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0), cube_coordinate(1), cube_coordinate(2) + 1),         INTERPOLATE(ratio(0)) * INTERPOLATE(ratio(1)) * ratio(2)));
    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1), cube_coordinate(2) + 1),     ratio(0) * INTERPOLATE(ratio(1)) * ratio(2)));

    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0), cube_coordinate(1) + 1, cube_coordinate(2)) + 1,     INTERPOLATE(ratio(0)) * ratio(1) * ratio(2)));
    weight_list.push_back(Eigen::Triplet<double>(row_index, get_index(cube_coordinate(0) + 1, cube_coordinate(1) + 1, cube_coordinate(2)) + 1, ratio.prod()));
  }

  W.setFromTriplets(weight_list.begin(), weight_list.end());
  // W.makeCompressed();
}
