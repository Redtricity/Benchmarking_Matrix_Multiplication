#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <cassert>
#include <immintrin.h>
#include "stats.hpp"

// $CXX -O3 -mavx matmul-assignment.cpp

#ifdef __PROSPERO__
// Only needed on the PS5: set to something sufficiently large.
unsigned int sceLibcHeapExtendedAlloc = 1; /* Switch to dynamic allocation */
size_t sceLibcHeapSize = SCE_LIBC_HEAP_SIZE_EXTENDED_ALLOC_NO_LIMIT; /* no upper limit for heap area */
#endif

#if (!defined(_MSC_VER))
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

struct mat
{
  float *data;
  const size_t sz;

  bool operator==(const mat &rhs) const
  {
    bool b_ret = true;
    const float tolerance = 0.1f;

    for (int i = 0; i < sz; i++) {
      for (int j = 0; j < sz; j++) {
        const float abs_diff = std::abs(this->data[i*sz+j] - rhs.data[i*sz+j]);
        b_ret = b_ret && (abs_diff < tolerance);
      }
    }

    return b_ret;
  }
};

void matmul(mat &mres, const mat &m1, const mat &m2)
{
  for (int i = 0; i < mres.sz; i++) {
    for (int j = 0; j < mres.sz; j++) {
      mres.data[i*mres.sz+j] = 0;
      for (int k = 0; k < mres.sz; k++) {
        mres.data[i*mres.sz+j] += m1.data[i*mres.sz+k] * m2.data[k*mres.sz+j];
      }
    }
  }
}

void matmul_simd(mat &mres, const mat &m1, const mat &m2) {
  // to do
}

void print_mat(const mat &m) {
  for (int i = 0; i < m.sz; i++) {
    for (int j = 0; j < m.sz; j++) {
      std::cout << std::setw(3) << m.data[i*m.sz+j] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

// A simple initialisation pattern. For an 8x8 matrix:

//  1   2   3   4   5   6   7   8
//  9  10  11  12  13  14  15  16
// 17  18  19  20  21  22  23  24
// 25  26  27  28  29  30  31  32
// 33  34  35  36  37  38  39  40
// 41  42  43  44  45  46  47  48
// 49  50  51  52  53  54  55  56
// 57  58  59  60  61  62  63  64

void init_mat(mat &m) {
  int count = 1;
  for (int i = 0; i < m.sz; i++) {
    for (int j = 0; j < m.sz; j++) {
      m.data[i*m.sz+j] = count++;
    }
  }
}

int main(int argc, char *argv[])
{
  unsigned int SZ = 1 << 3; // (1 << 10) == 1024
  // n.b. these calls to new have no alignment specifications
  mat mres{new float[SZ*SZ],SZ},m{new float[SZ*SZ],SZ},id{new float[SZ*SZ],SZ};
  mat mres_simd{new float[SZ*SZ],SZ};
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t t1, t2;

  std::cout << "Each " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(float)*SZ*SZ << " bytes.\n";

  init_mat(m);

  t1 = high_resolution_clock::now();
  matmul(mres,m,m);
  t2 = high_resolution_clock::now();

  auto d = duration_cast<microseconds>(t2-t1).count();
  std::cout << d << ' ' << "microseconds.\n";

  t1 = high_resolution_clock::now();
  matmul_simd(mres_simd,m,m);
  t2 = high_resolution_clock::now();

  d = duration_cast<microseconds>(t2-t1).count();
  std::cout << d << ' ' << "microseconds.\n";

  print_mat(m);
  print_mat(m);
  print_mat(mres);

  const bool correct = mres_simd==mres;
  //assert(correct); // uncomment when you have implemented matmul_simd

  delete [] mres.data;
  delete [] mres_simd.data;
  delete [] m.data;
  delete [] id.data;
  return correct ? 0 : -1;
}
