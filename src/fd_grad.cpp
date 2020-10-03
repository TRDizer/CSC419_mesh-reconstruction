#include "fd_grad.h"
#include "fd_partial_derivative.h"
#include "igl/cat.h"
#include <iostream>

void fd_grad(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  Eigen::SparseMatrix<double> & G)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  ////////////////////////////////////////////////////////////////////////////
	Eigen::SparseMatrix<double> Dx, Dy, Dz;
  fd_partial_derivative(nx, ny, nz,h, 0, Dx);
  fd_partial_derivative(nx, ny, nz,h, 1, Dy);
  fd_partial_derivative(nx, ny, nz,h, 2, Dz);

  // Attempted to use coeffRef to populate G from ground up but I was again stuck on assertion failure
  // Reference Sarah's approach on using igl::cat
  Eigen::SparseMatrix<double> temp((nx - 1) * ny * nz + nx * (ny - 1) * nz, nx * ny * nz);
  G.resize(temp.size() + nx * ny * (nz - 1), nx * ny * nz);
  igl::cat(1, Dx, Dy, temp);
  igl::cat(1, temp, Dz, G);
}
