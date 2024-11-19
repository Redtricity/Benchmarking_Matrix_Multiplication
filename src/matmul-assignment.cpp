#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <thread> // Multithreading library
#include <vector> // Vecotr library
#include <cassert> // Assert
#include <immintrin.h> // SIMD
#include <numeric>
#include <cmath>
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

// Represents a Matrix - LM
struct mat
{
  double *data;
  const size_t sz; // Matrix size - LM

  bool operator==(const mat &rhs) const
  {
    bool b_ret = true;
    const double tolerance = 0.1f;

    // Checks if two matrices are equal - LM
    for (int i = 0; i < sz; i++) {
      for (int j = 0; j < sz; j++) {
        const double abs_diff = std::abs(this->data[i*sz+j] - rhs.data[i*sz+j]);
        b_ret = b_ret && (abs_diff < tolerance);
      }
    }

    return b_ret;
  }
};

void Transpose(const mat &m2, mat &mres, mat m2t) {
    for (int i = 0; i < m2t.sz; i++) {
        for (int j = 0; j < m2t.sz; j++) {
            m2t.data[i * mres.sz + j] = m2.data[j * mres.sz + i];
        }
    }
}
// Simple multiplication function - LM
void matmul(mat &mres, const mat &m1, const mat &m2)
{
    // Multiplies and adds two Matrices
  for (int i = 0; i < mres.sz; i++) { // Rows - LM
    for (int j = 0; j < mres.sz; j++) { // Collumns - LM
      mres.data[i*mres.sz+j] = 0;
      for (int k = 0; k < mres.sz; k++) { // Inner - LM
        mres.data[i*mres.sz+j] += m1.data[i*mres.sz+k] * m2.data[k*mres.sz+j]; // Multiplication and Addition - LM
      }
    }
  }
}

//// Parallel Matrix Multiplication (Original without simd added) - LM
void matmul_parallel(mat& mres, const mat& m1, const mat& m2, int num_threads) {
    int rows_per_thread = mres.sz / num_threads; //  8x8 Matrix / 4 threads meaning 1 thread deals with 2 rows

    std::vector<std::thread> threads; // vector to hold the threads - LM (note - maybe use pthread)

    for (int t = 0; t < num_threads; t++) { // Identifys threads to divide the matrix into rows -LM
        int start_row = t * rows_per_thread; // Thread Start row - LM 
        int end_row = (t == num_threads - 1) ? mres.sz : (t + 1) * rows_per_thread; // Thread End Row - LM

        threads.push_back(std::thread([=, &mres, &m1, &m2]() {
            for (int i = start_row; i < end_row; i++) { // Rows - LM
                for (int j = 0; j < mres.sz; j++) { // Collumns - LM
                    mres.data[i * mres.sz + j] = 0;
                    for (int k = 0; k < mres.sz; k++) { // Inner - LM
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

// Parallel Matrix with SIMD Multiplication - LM
// Double
void matmul_parallel_simd_double(mat& mres, const mat& m1, const mat& m2, int num_threads) {
    int rows_per_thread = mres.sz / num_threads; //  8x8 Matrix / 4 threads meaning 1 thread deals with 2 rows

    std::vector<std::thread> threads; // vector to hold the threads - LM (note - maybe use pthread)

    mat m2t{ new double[m2.sz * m2.sz], m2.sz };
    Transpose(m2, mres, m2t);

    for (int t = 0; t < num_threads; t++) { // Identifys threads to divide the matrix into rows -LM
        int start_row = t * rows_per_thread; // Thread Start row - LM 
        int end_row = (t == num_threads - 1) ? mres.sz : (t + 1) * rows_per_thread; // Thread End Row - LM

        //Creates threads to push to the vector
        threads.push_back(std::thread([=, &mres, &m1, &m2t]() {
            for (int i = start_row; i < end_row; i++) { // Rows - LM
                for (int j = 0; j < mres.sz; j++) { // Collumns - LM
                    __m256d result = _mm256_setzero_pd(); // Initilses the simd register
                    //int counter = 1;

                    for (int k = 0; k < mres.sz; k += 4) {  // Go through the Matrix in 4s (SIMD can handle 4 flots at once)
                        __m256d m1_values = _mm256_load_pd(&m1.data[i * mres.sz + k]); // Load 4 numbers from row i of matrix m1 into a SIMD register
                        __m256d m2_values = _mm256_loadu_pd(&m2t.data[j * mres.sz + k]); // Load 4 numners from collum J of matrix m2 into another SIMD register

                        __m256d product = _mm256_mul_pd(m1_values, m2_values); // Multiply the values

                        result = _mm256_add_pd(result, product); // Add the result to the current total
                    }

                    double res[4]; // Store the 4 numbers in result back in an array
                    _mm256_store_pd(res, result); // Move the SIMD values into an array called res
                    mres.data[i * mres.sz + j] = res[0] + res[1] + res[2] + res[3]; //Adds the 4 numbers in res to get one number for this position in mres
                }
            }

            }));
    }

    // Joins all the threads
    for (auto& thread : threads) {
        thread.join();
    }
}

// SIMD Multiplication - LM
// Double
void matmul_simd_double(mat& mres, const mat& m1, const mat& m2) {
    mat m2t{ new double[m2.sz * m2.sz], m2.sz };
    Transpose(m2, mres, m2t);

    // to do
    for (int i = 0; i < mres.sz; i++) { // Go through each rows - LM
        for (int j = 0; j < mres.sz; j++) { // Go through each collumn - LM
            __m256d result = _mm256_setzero_pd(); // Create SIMD Variable that has a starting value of 0
            //int counter = 1;
            for (int k = 0; k < mres.sz; k += 4) {  // Go through the Matrix in 4s (SIMD can handle 4 flots at once)
                __m256d m1_values = _mm256_load_pd(&m1.data[i * mres.sz + k]); // Load 4 numbers from row i of matrix m1 into a SIMD register
                __m256d m2_values = _mm256_loadu_pd(&m2t.data[j * mres.sz + k]); // Load 4 numners from collum J of matrix m2 into another SIMD register

                __m256d product = _mm256_mul_pd(m1_values, m2_values); // Multiply the values

                result = _mm256_add_pd(result, product); // Add the result to the current total 
            }          

           double res[4]; // Store the 4 numbers in result back in an array
           _mm256_store_pd(res, result); // Move the SIMD values into an array called res
           mres.data[i * mres.sz + j] = res[0] + res[1] + res[2] + res[3]; //Adds the 4 numbers in res to get one number for this position in mres
        } 
    }
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
        m.data[i * m.sz + j] = count++;
        if (count == 1)
            count = 1;
    }
  }
}

double standard_deviation(const std::vector<double>& data)
{
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sum_sqr_diffs = 0.0;
    for (const auto& x : data) {
        sum_sqr_diffs += (x - mean) * (x - mean);
    }
    double variance = sum_sqr_diffs / (data.size() - 1);
    return std::sqrt(variance);
}


int main(int argc, char *argv[])
{
  std::vector<double> data(6);
  unsigned int SZ = 1 << 3; // (1 << 10) == 1024 (Matrix size is 8 - LM)
  // n.b. these calls to new have no alignment specifications
  mat mres{new double[SZ*SZ],SZ},m{new double[SZ*SZ],SZ},id{new double[SZ*SZ],SZ};
  mat mres_parallel{ new double[SZ * SZ],SZ };
  mat mres_parallel_simd{ new double[SZ * SZ],SZ };
  mat mres_simd{new double[SZ*SZ],SZ};
  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t t1, t2;

  // Matrix description
  std::cout << "Each " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(double)*SZ*SZ << " bytes.\n";

  init_mat(m); // Initialise Matrix with sample values - LM

  // Timing simple multiplication - LM
  t1 = high_resolution_clock::now();
  matmul(mres,m,m); // Simple Multiplication - LM
  t2 = high_resolution_clock::now();

  auto d = duration_cast<microseconds>(t2-t1).count();
  std::cout << "Simple multiplication time: " << d << ' ' << "microseconds.\n";

  // Timing SIMD - LM
  for (int i = 0; i <= 5; i++) {

      t1 = high_resolution_clock::now();
      matmul_simd_double(mres_simd, m, m); // Simd Multiplication - LM
      t2 = high_resolution_clock::now();

      double d = duration_cast<microseconds>(t2 - t1).count();
      data[i] = d;
  }
  std::cout << standard_deviation(data) << "\n";
  //std::cout << "Simd multiplication time: " << d << ' ' << "microseconds.\n";

  // Timing Parallel Multiplication (Original) - LM
  int num_threads = 4; // Num of threads - LM
  t1 = high_resolution_clock::now();
  matmul_parallel(mres_parallel, m, m, num_threads); // Parallel Multiplication - LM
  t2 = high_resolution_clock::now();

  d = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "Parallel multiplication time with " << num_threads << " threads: " << d << ' ' << "microseconds.\n";

  // Timing Parallel Multiplication with simd - LM
  //int num_threads = 4; // Duplicate - LM
  t1 = high_resolution_clock::now();
  matmul_parallel_simd_double(mres_parallel_simd, m, m, num_threads); // Parallel Multiplication - LM
  t2 = high_resolution_clock::now();

  d = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "Parallel simd multiplication time with " << num_threads << " threads: " << d << ' ' << "microseconds.\n";
  
  //Prints the matrix results - LM

  std::cout << "Initial Matrix:" << std::endl;
  //print_mat(m); // Print simple multiplication

  std::cout << "Simple multiplication:" << std::endl;
  //print_mat(mres);

  std::cout << "Simd:" << std::endl;
  //print_mat(mres_simd); // Print result of simd - LM

  std::cout << "Parallel:" << std::endl;
  //print_mat(mres_parallel); // Print result of parallel multiplication - LM

  std::cout << "Parallel with Simd:" << std::endl;
 //print_mat(mres_parallel_simd); // Print result of parallel with simd multiplication - LM


  // Assert statement to check if both multiplications are the same.
  bool b1 = mres == mres_parallel;
  bool b2 = mres == mres_simd;
  bool b3 = mres == mres_parallel;
  bool b4 = mres == mres_parallel_simd;

  assert(b1);
  assert(b2);
  assert(b3); // ok
  assert(b4);

  delete [] mres.data;
  delete [] mres_parallel.data;
  delete [] mres_parallel_simd.data;
  delete [] mres_simd.data;
  delete [] m.data;
  delete [] id.data;
  return  0;
}
