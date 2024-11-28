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

  float* s_data;
  const size_t S_sz;

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

#pragma region Simple Multiplication
// Single precision
void S_matmul(mat& S_mres, const mat& S_m1, const mat& S_m2)
{
    // Multiplies and adds two Matrices
    for (int i = 0; i < S_mres.S_sz; i++) { // Rows - LM
        for (int j = 0; j < S_mres.S_sz; j++) { // Collumns - LM
            S_mres.s_data[i * S_mres.S_sz + j] = 0;
            for (int k = 0; k < S_mres.S_sz; k++) { // Inner - LM
                S_mres.s_data[i * S_mres.S_sz + j] += S_m1.s_data[i * S_mres.S_sz + k] * S_m2.s_data[k * S_mres.S_sz + j]; // Multiplication and Addition - LM
            }
        }
    }
}

// Double precision
void matmul(mat& mres, const mat& m1, const mat& m2)
{
    // Multiplies and adds two Matrices
    for (int i = 0; i < mres.sz; i++) { // Rows - LM
        for (int j = 0; j < mres.sz; j++) { // Collumns - LM
            mres.data[i * mres.sz + j] = 0;
            for (int k = 0; k < mres.sz; k++) { // Inner - LM
                mres.data[i * mres.sz + j] += m1.data[i * mres.sz + k] * m2.data[k * mres.sz + j]; // Multiplication and Addition - LM
            }
        }
    }
}
#pragma endregion

#pragma region Parallel Matrix Multiplication
//// Single precision
void S_matmul_parallel(mat& S_mres, const mat& S_m1, const mat& S_m2, int num_threads) {
    int rows_per_thread = S_mres.S_sz / num_threads; //  8x8 Matrix / 4 threads meaning 1 thread deals with 2 rows

    std::vector<std::thread> threads; // vector to hold the threads - LM (note - maybe use pthread)

    for (int t = 0; t < num_threads; t++) { // Identifys threads to divide the matrix into rows -LM
        int start_row = t * rows_per_thread; // Thread Start row - LM 
        int end_row = (t == num_threads - 1) ? S_mres.S_sz : (t + 1) * rows_per_thread; // Thread End Row - LM

        threads.push_back(std::thread([=, &S_mres, &S_m1, &S_m2]() {
            for (int i = start_row; i < end_row; i++) { // Rows - LM
                for (int j = 0; j < S_mres.S_sz; j++) { // Collumns - LM
                    S_mres.s_data[i * S_mres.S_sz + j] = 0;
                    for (int k = 0; k < S_mres.S_sz; k++) { // Inner - LM
                        S_mres.s_data[i * S_mres.S_sz + j] += S_m1.s_data[i * S_mres.S_sz + k] * S_m2.s_data[k * S_mres.S_sz + j];
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

//// Double Precision
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
#pragma endregion

#pragma region Parallel Matrix with SIMD Multiplication
// Single precision
void matmul_parallel_simd(mat& S_mres, const mat& m1, const mat& m2, int num_threads) {
    int rows_per_thread = S_mres.S_sz / num_threads; //  8x8 Matrix / 4 threads meaning 1 thread deals with 2 rows

    std::vector<std::thread> threads; // vector to hold the threads - LM (note - maybe use pthread)

    for (int t = 0; t < num_threads; t++) { // Identifys threads to divide the matrix into rows -LM
        int start_row = t * rows_per_thread; // Thread Start row - LM 
        int end_row = (t == num_threads - 1) ? S_mres.S_sz : (t + 1) * rows_per_thread; // Thread End Row - LM

        //Creates threads to push to the vector
        threads.push_back(std::thread([=, &S_mres, &m1, &m2]() {
            for (int i = start_row; i < end_row; i++) { // Rows - LM
                for (int j = 0; j < S_mres.S_sz; j++) { // Collumns - LM
                    __m128 result = _mm_setzero_ps(); // Initilses the simd register

                    for (int k = 0; k < S_mres.S_sz; k += 4) {  // Go through the Matrix in 4s (SIMD can handle 4 flots at once)
                        __m128 m1_values = _mm_load_ps(&m1.s_data[i * S_mres.S_sz + k]); // Load 4 numbers from row i of matrix m1 into a SIMD register
                        __m128 m2_values = _mm_loadu_ps(&m2.s_data[k * S_mres.S_sz + j]); // Load 4 numners from collum J of matrix m2 into another SIMD register

                        __m128 product = _mm_mul_ps(m1_values, m2_values); // Multiply the values

                        result = _mm_add_ps(result, product); // Add the result to the current total
                    }
                    float res[4]; // Store the 4 numbers in result back in an array
                    _mm_storeu_ps(res, result); // Move the SIMD values into an array called res
                    S_mres.s_data[i * S_mres.S_sz + j] = res[0] + res[1] + res[2] + res[3]; //Adds the 4 numbers in res to get one number for this position in mres
                }
            }

            }));
    }

    // Joins all the threads
    for (auto& thread : threads) {
        thread.join();
    }
}

// Double precision
void matmul_parallel_simd(mat& mres, const mat& m1, const mat& m2, int num_threads) {
    int rows_per_thread = mres.sz / num_threads; //  8x8 Matrix / 4 threads meaning 1 thread deals with 2 rows

    std::vector<std::thread> threads; // vector to hold the threads - LM (note - maybe use pthread)

    mat m2t{ new double[m2.sz * m2.sz], m2.sz };
    for (int i = 0; i < m2t.sz; i++) {
        for (int j = 0; j < m2t.sz; j++) {
            m2t.data[i * mres.sz + j] = m2.data[j * mres.sz + i]; // ??
        }
    }

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

                        /*__m256d m2_values = _mm256_set_pd(
                            m2.data[(k + 3) * mres.sz + j],
                            m2.data[(k + 2) * mres.sz + j],
                            m2.data[(k + 1) * mres.sz + j],
                            m2.data[k * mres.sz + j]
                        );*/
                        //__m128d m2_d_values = _mm_cvtps_pd(m2_values);

                        __m256d product = _mm256_mul_pd(m1_values, m2_values); // Multiply the values

                        result = _mm256_add_pd(result, product); // Add the result to the current total
                    }

                    double res[4]; // Store the 4 numbers in result back in an array
                    _mm256_store_pd(res, result); // Move the SIMD values into an array called res
                    mres.data[i * mres.sz + j] = res[0] + res[1] + res[2] + res[3]; //Adds the 4 numbers in res to get one number for this position in mres

                    /*double res1[2]; // Store the first 2 numbers in result back in an array
                    double res2[2]; // Store the last 2 numbers in res
                    if (counter >= 2) {
                        _mm_store_pd(res1, result); // Move the SIMD values into an array called res
                    }
                    else {
                       _mm_store_pd(res2, result);
                       if (counter == 4) {
                           counter = 1;
                       }
                    }
                    counter++;

                    mres.data[i * mres.sz + j] = res1[0] + res1[1] + res2[0] + res2[1]; //Adds the 4 numbers in res to get one number for this position in mres */
                }
            }

            }));
    }

    // Joins all the threads
    for (auto& thread : threads) {
        thread.join();
    }
}
#pragma endregion

#pragma region SIMD Multiplication
// Single precision
void S_matmul_simd(mat& S_mres, const mat& m1, const mat& m2) {
    // to do
    for (int i = 0; i < S_mres.S_sz; i++) { // Go through each rows - LM
        for (int j = 0; j < S_mres.S_sz; j++) { // Go through each collumn - LM
            __m128 result = _mm_setzero_ps(); // Create SIMD Variable that has a starting value of 0

            for (int k = 0; k < S_mres.S_sz; k += 4) {  // Go through the Matrix in 4s (SIMD can handle 4 flots at once)
                __m128 m1_values = _mm_load_ps(&m1.s_data[i * S_mres.S_sz + k]); // Load 4 numbers from row i of matrix m1 into a SIMD register
                //__m128 m2_values = _mm_loadu_ps(&m2.data[k * mres.sz + j]); // Load 4 numners from collum J of matrix m2 into another SIMD register

                __m128 m2_values = _mm_set_ps(
                    m2.s_data[(k + 3) * S_mres.S_sz + j],
                    m2.s_data[(k + 2) * S_mres.S_sz + j],
                    m2.s_data[(k + 1) * S_mres.S_sz + j],
                    m2.s_data[k * S_mres.S_sz + j]
                );

                __m128 product = _mm_mul_ps(m1_values, m2_values); // Multiply the values

                result = _mm_add_ps(result, product); // Add the result to the current total 
            }
            float res[4]; // Store the 4 numbers in result back in an array
            _mm_storeu_ps(res, result); // Move the SIMD values into an array called res
            S_mres.s_data[i * S_mres.S_sz + j] = res[0] + res[1] + res[2] + res[3]; //Adds the 4 numbers in res to get one number for this position in mres

            // Debug: print the result for each element
            //std::cout << "mres[" << i << "][" << j << "] = " << mres.data[i * mres.sz + j] << std::endl;
        }
    }
}

// Double precision
void matmul_simd(mat& mres, const mat& m1, const mat& m2) {

    mat m2t{ new double[m2.sz * m2.sz], m2.sz };
    for (int i = 0; i < m2t.sz; i++) {
        for (int j = 0; j < m2t.sz; j++) {
            m2t.data[i * mres.sz + j] = m2.data[j * mres.sz + i]; // ??
        }
    }

    // to do
    for (int i = 0; i < mres.sz; i++) { // Go through each rows - LM
        for (int j = 0; j < mres.sz; j++) { // Go through each collumn - LM
            __m256d result = _mm256_setzero_pd(); // Create SIMD Variable that has a starting value of 0
            //int counter = 1;
            for (int k = 0; k < mres.sz; k += 4) {  // Go through the Matrix in 4s (SIMD can handle 4 flots at once)
                __m256d m1_values = _mm256_load_pd(&m1.data[i * mres.sz + k]); // Load 4 numbers from row i of matrix m1 into a SIMD register
                __m256d m2_values = _mm256_loadu_pd(&m2t.data[j * mres.sz + k]); // Load 4 numners from collum J of matrix m2 into another SIMD register

                /*__m256d m2_values = _mm256_set_pd(
                    m2.data[(k + 3) * mres.sz + j],
                    m2.data[(k + 2) * mres.sz + j],
                    m2.data[(k + 1) * mres.sz + j],
                    m2.data[k * mres.sz + j]
                );*/
                //converts m2 values to a __m128d (i couldnt find a double equivalent for set ps that took 4 arguments)
                //__m128d m2_d_values = _mm_cvtps_pd(m2_values); 
                __m256d product = _mm256_mul_pd(m1_values, m2_values); // Multiply the values

                result = _mm256_add_pd(result, product); // Add the result to the current total 
            }

            /*double res1[2]; // Store the first 2 numbers in result back in an array
            double res2[2]; // Store the last 2 numbers in res
            if (counter >= 2) {
                _mm_store_pd(res1, result); // Move the SIMD values into an array called res
            }
            else {
                _mm_store_pd(res2, result);
                if (counter == 4) {
                    counter = 1;
                }
            }
            counter++;*/

            // mres.data[i * mres.sz + j] = res1[0] + res1[1] + res2[0] + res2[1]; //Adds the 4 numbers in res to get one number for this position in mres

            double res[4]; // Store the 4 numbers in result back in an array
            _mm256_store_pd(res, result); // Move the SIMD values into an array called res
            mres.data[i * mres.sz + j] = res[0] + res[1] + res[2] + res[3]; //Adds the 4 numbers in res to get one number for this position in mres

            // Debug: print the result for each element
            //std::cout << "mres[" << i << "][" << j << "] = " << mres.data[i * mres.sz + j] << std::endl;
        }
    }
}
#pragma endregion


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

int main(int argc, char *argv[])
{
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
  t1 = high_resolution_clock::now();
  matmul_simd(mres_simd,m,m); // Simd Multiplication - LM
  t2 = high_resolution_clock::now();

  d = duration_cast<microseconds>(t2-t1).count();
  std::cout << "Simd multiplication time: " << d << ' ' << "microseconds.\n";

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
  matmul_parallel_simd(mres_parallel_simd, m, m, num_threads); // Parallel Multiplication - LM
  t2 = high_resolution_clock::now();

  d = duration_cast<microseconds>(t2 - t1).count();
  std::cout << "Parallel simd multiplication time with " << num_threads << " threads: " << d << ' ' << "microseconds.\n";
  
  //Prints the matrix results - LM

  std::cout << "Initial Matrix:" << std::endl;
  print_mat(m); // Print simple multiplication

  std::cout << "Simple multiplication:" << std::endl;
  print_mat(mres);

  std::cout << "Simd:" << std::endl;
  print_mat(mres_simd); // Print result of simd - LM

  std::cout << "Parallel:" << std::endl;
  print_mat(mres_parallel); // Print result of parallel multiplication - LM

  std::cout << "Parallel with Simd:" << std::endl;
  print_mat(mres_parallel_simd); // Print result of parallel with simd multiplication - LM


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
