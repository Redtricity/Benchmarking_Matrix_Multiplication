#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <cassert>
#include <immintrin.h>
#include "stats.hpp"
#include <thread> // Multithreading library
#include <vector> // Vecotr library
#include <cassert> // Assert

// $CXX -O3 -mavx matmul-assignment.cpp

#ifdef __PROSPERO__
// Only needed on the PS5: set to something sufficiently large.
unsigned int sceLibcHeapExtendedAlloc = 1; /* Switch to dynamic allocation */
size_t sceLibcHeapSize = SCE_LIBC_HEAP_SIZE_EXTENDED_ALLOC_NO_LIMIT; /* no upper limit for heap area */
#endif

#if (!defined(_MSC_VER))
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

// Represents a Matrix - LM
struct mat
{
  float *data;
  const size_t sz; // Matrix size - LM

  bool operator==(const mat &rhs) const
  {
    bool b_ret = true;
    const float tolerance = 0.1f;

    // Checks if two matrices are equal - LM
    for (int i = 0; i < sz; i++) {
      for (int j = 0; j < sz; j++) {
        const float abs_diff = std::abs(this->data[i*sz+j] - rhs.data[i*sz+j]);
        b_ret = b_ret && (abs_diff < tolerance);
      }
    }

    return b_ret;
  }
};

// Multiplication function - LM
void matmul(mat &mres, const mat &m1, const mat &m2)
{
  for (int i = 0; i < mres.sz; i++) { // Rows - LM
    for (int j = 0; j < mres.sz; j++) { // Collumns - LM
      mres.data[i*mres.sz+j] = 0;
      for (int k = 0; k < mres.sz; k++) {
        mres.data[i*mres.sz+j] += m1.data[i*mres.sz+k] * m2.data[k*mres.sz+j]; // Multiplication and Addition - LM
      }
    }
  }
}

// Parallel Matrix Multiplication - LM
void matmul_parallel(mat& mres, const mat& m1, const mat& m2, int num_threads) {
    int rows_per_thread = mres.sz / num_threads; //  8x8 Matrix / 4 threads meaning 1 thread deals with 2 rows

    std::vector<std::thread> threads; // vector to hold the threads - LM (note - maybe use pthread)

    for (int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread; // Thread Start row
        int end_row = (t == num_threads - 1) ? mres.sz : (t + 1) * rows_per_thread; // Thread End Row

        threads.push_back(std::thread([=, &mres, &m1, &m2]() {
            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < mres.sz; j++) {
                    mres.data[i * mres.sz + j] = 0;
                    for (int k = 0; k < mres.sz; k++) {
                        mres.data[i * mres.sz + j] += m1.data[i * mres.sz + k] * m2.data[k * mres.sz + j];
                    }
                }
            }
            }));
    }

    // Joins all the threads
    for (auto& thread : threads) {
        thread.join();
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

// Test multiplication function
int main(int argc, char *argv[])
{
  unsigned int SZ = 1 << 3; // (1 << 10) == 1024 (Matrix size is 8 - LM)
  // n.b. these calls to new have no alignment specifications
  mat mres{new float[SZ*SZ],SZ},m{new float[SZ*SZ],SZ},id{new float[SZ*SZ],SZ};
  mat mres_parallel{ new float[SZ * SZ],SZ };
  mat mres_simd{new float[SZ*SZ],SZ};
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t t1, t2;

  // Matrix description
  std::cout << "Each " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(float)*SZ*SZ << " bytes.\n";

  init_mat(m); // Initialise Matrix with sample values - LM

  // Timing simple multiplication - LM
  t1 = high_resolution_clock::now();
  matmul(mres,m,m);
  t2 = high_resolution_clock::now();

  auto d = duration_cast<microseconds>(t2-t1).count();
  std::cout << "Simple multiplication time: " << d << ' ' << "microseconds.\n";

  t1 = high_resolution_clock::now();
  matmul_simd(mres_simd,m,m);
  t2 = high_resolution_clock::now();

  d = duration_cast<microseconds>(t2-t1).count();
  std::cout << " simd" << d << ' ' << "microseconds.\n";

  // Timing Parallel Multiplication - LM
  int num_threads = 4; // Num of threads - LM
  t1 = high_resolution_clock::now();
  matmul_parallel(mres_parallel, m, m, num_threads);
  t2 = high_resolution_clock::now();

  d = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "Parallel multiplication with " << num_threads << " threads: " << d << ' ' << "microseconds.\n";
  
  // Assert statement to check if both multiplications are the same.
  assert(mres == mres_parallel);

  print_mat(m); // Print simple multiplication
  print_mat(m);
  print_mat(mres_parallel); // Print result of parallel multiplication


  const bool correct = mres_simd==mres;
  //assert(correct); // uncomment when you have implemented matmul_simd

  delete [] mres.data;
  delete [] mres_parallel.data;
  delete [] mres_simd.data;
  delete [] m.data;
  delete [] id.data;
  return correct ? 0 : -1;
}
