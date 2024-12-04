#include <iostream>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <thread> 
#include <vector> 
#include <cassert> 
#include <immintrin.h> 
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

// Written by Lara Mcintyre B00418895 and  Chinedum Vincent-Eloagu B00409207

// Represents the Double Precision Matrix 
struct mat
{
  double *data;
  const size_t sz; 

  bool operator==(const mat &rhs) const
  {
    bool b_ret = true;
    const double tolerance = 0.1f;

    // Checks if two matrices are equal
    for (int i = 0; i < sz; i++) {
      for (int j = 0; j < sz; j++) {
        const double abs_diff = std::abs(this->data[i*sz+j] - rhs.data[i*sz+j]);
        b_ret = b_ret && (abs_diff < tolerance);
      }
    }

    return b_ret;
  }
};
// Represents the Single Precision Matrix 
struct Smat
{
    float* s_data;
    const size_t S_sz; // Matrix size 

    bool operator==(const Smat& rhs) const
    {
        bool b_ret = true;
        const float tolerance = 0.1f;

        // Checks if two matrices are equal 
        for (int i = 0; i < S_sz; i++) {
            for (int j = 0; j < S_sz; j++) {
                const float abs_diff = std::abs(this->s_data[i * S_sz + j] - rhs.s_data[i * S_sz + j]);
                b_ret = b_ret && (abs_diff < tolerance);
            }
        }

        return b_ret;
    }
};

// Transpose for Double Precision
void Transpose_Double(const mat &m2, mat &mres, mat m2t) {
    for (int i = 0; i < m2t.sz; i++) {
        for (int j = 0; j < m2t.sz; j++) {
            m2t.data[i * mres.sz + j] = m2.data[j * mres.sz + i];
        }
    }
}
// Transpose for Single Precision
void Transpose_Single(const Smat& m2, Smat& S_mres, Smat m2t) {
    for (int i = 0; i < m2t.S_sz; i++) {
        for (int j = 0; j < m2t.S_sz; j++) {
            m2t.s_data[i * S_mres.S_sz + j] = m2.s_data[j * S_mres.S_sz + i];
        }
    }
}

#pragma region Simple multiplication

//Single
void S_matmul(Smat& S_mres, const Smat& S_m1, const Smat& S_m2)
{
    // Multiplies and adds two Matrices
    for (int i = 0; i < S_mres.S_sz; i++) { // Rows 
        for (int j = 0; j < S_mres.S_sz; j++) { // Collumns 
            S_mres.s_data[i * S_mres.S_sz + j] = 0;
            for (int k = 0; k < S_mres.S_sz; k++) { // Inner 
                S_mres.s_data[i * S_mres.S_sz + j] += S_m1.s_data[i * S_mres.S_sz + k] * S_m2.s_data[k * S_mres.S_sz + j]; // Multiplication and Addition 
            }
        }
    }
}

// Double
void matmul(mat& mres, const mat& m1, const mat& m2)
{
    for (int i = 0; i < mres.sz; i++) { 
        for (int j = 0; j < mres.sz; j++) { 
            mres.data[i * mres.sz + j] = 0;
            for (int k = 0; k < mres.sz; k++) { 
                mres.data[i * mres.sz + j] += m1.data[i * mres.sz + k] * m2.data[k * mres.sz + j];
            }
        }
    }
}

#pragma endregion

#pragma region Parallel Multiplication

//Single
void S_matmul_parallel(Smat& S_mres, const Smat& S_m1, const Smat& S_m2, int num_threads) {
    int rows_per_thread = S_mres.S_sz / num_threads; //  8x8 Matrix / 4 threads meaning 1 thread deals with 2 rows

    std::vector<std::thread> threads; // vector to hold the threads

    for (int t = 0; t < num_threads; t++) { // Identifys threads to divide the matrix into rows 
        int start_row = t * rows_per_thread; // Thread Start row  
        int end_row = (t == num_threads - 1) ? S_mres.S_sz : (t + 1) * rows_per_thread; // Thread End Row 

        threads.push_back(std::thread([=, &S_mres, &S_m1, &S_m2]() {
            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < S_mres.S_sz; j++) {  
                    S_mres.s_data[i * S_mres.S_sz + j] = 0;
                    for (int k = 0; k < S_mres.S_sz; k++) { 
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

//Double
void matmul_parallel(mat& mres, const mat& m1, const mat& m2, int num_threads) {
    int rows_per_thread = mres.sz / num_threads; 

    std::vector<std::thread> threads; 

    for (int t = 0; t < num_threads; t++) { 
        int start_row = t * rows_per_thread; 
        int end_row = (t == num_threads - 1) ? mres.sz : (t + 1) * rows_per_thread; 

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

    for (auto& thread : threads) {
        thread.join();
    }
}

#pragma endregion

#pragma region Parallel with SIMD Multiplication

//Single
void matmul_parallel_simd(Smat& S_mres, const Smat& m1, const Smat& m2, int num_threads) {
    int rows_per_thread = S_mres.S_sz / num_threads; 

    std::vector<std::thread> threads; 

    Smat m2t{ new float[m2.S_sz * m2.S_sz], m2.S_sz };
    Transpose_Single(m2, S_mres, m2t);

    for (int t = 0; t < num_threads; t++) { 
        int start_row = t * rows_per_thread; 
        int end_row = (t == num_threads - 1) ? S_mres.S_sz : (t + 1) * rows_per_thread; 

        //Creates threads to push to the vector
        threads.push_back(std::thread([=, &S_mres, &m1, &m2]() {
            for (int i = start_row; i < end_row; i++) { 
                for (int j = 0; j < S_mres.S_sz; j++) { 
                    __m128 result = _mm_setzero_ps(); // Initilses the simd register

                    for (int k = 0; k < S_mres.S_sz; k += 4) {  // Go through the Matrix in 4s (SIMD can handle 4 flots at once)
                        __m128 m1_values = _mm_loadu_ps(&m1.s_data[i * S_mres.S_sz + k]); // Load 4 numbers from row i of matrix m1 into a SIMD register
                        __m128 m2_values = _mm_loadu_ps(&m2t.s_data[j * S_mres.S_sz + k]); // Load 4 numners from collum J of matrix m2 into another SIMD register

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

// Double
void matmul_parallel_simd_double(mat& mres, const mat& m1, const mat& m2, int num_threads) {
    int rows_per_thread = mres.sz / num_threads; 
    std::vector<std::thread> threads; 

    mat m2t{ new double[m2.sz * m2.sz], m2.sz };
    Transpose_Double(m2, mres, m2t);

    for (int t = 0; t < num_threads; t++) { 
        int start_row = t * rows_per_thread; 
        int end_row = (t == num_threads - 1) ? mres.sz : (t + 1) * rows_per_thread; 

        
        threads.push_back(std::thread([=, &mres, &m1, &m2t]() {
            for (int i = start_row; i < end_row; i++) { 
                for (int j = 0; j < mres.sz; j++) { 
                    __m256d result = _mm256_setzero_pd(); 

                    for (int k = 0; k < mres.sz; k += 4) {  
                        __m256d m1_values = _mm256_load_pd(&m1.data[i * mres.sz + k]); 
                        __m256d m2_values = _mm256_loadu_pd(&m2t.data[j * mres.sz + k]); 

                        __m256d product = _mm256_mul_pd(m1_values, m2_values); 

                        result = _mm256_add_pd(result, product); 
                    }

                    double res[4]; 
                    _mm256_store_pd(res, result); 
                    mres.data[i * mres.sz + j] = res[0] + res[1] + res[2] + res[3]; 
                }
            }

            }));
    }

   
    for (auto& thread : threads) {
        thread.join();
    }
}

#pragma endregion

#pragma region SIMD Multiplication

//Single
void S_matmul_simd(Smat& S_mres, const Smat& m1, const Smat& m2) {
    Smat m2t{ new float[m2.S_sz * m2.S_sz], m2.S_sz };
    Transpose_Single(m2, S_mres, m2t);
    for (int i = 0; i < S_mres.S_sz; i++) { 
        for (int j = 0; j < S_mres.S_sz; j++) { 
            __m128 result = _mm_setzero_ps(); // Create SIMD Variable that has a starting value of 0

            for (int k = 0; k < S_mres.S_sz; k += 4) {  // Go through the Matrix in 4s (SIMD can handle 4 flots at once)
                __m128 m1_values = _mm_load_ps(&m1.s_data[i * S_mres.S_sz + k]); // Load 4 numbers from row i of matrix m1 into a SIMD register
                __m128 m2_values = _mm_loadu_ps(&m2t.s_data[j * S_mres.S_sz + k]); // Load 4 numners from collum J of matrix m2 into another SIMD register
                
                __m128 product = _mm_mul_ps(m1_values, m2_values); // Multiply the values

                result = _mm_add_ps(result, product); // Add the result to the current total 
            }
            float res[4]; // Store the 4 numbers in result back in an array
            _mm_storeu_ps(res, result); // Move the SIMD values into an array called res
            S_mres.s_data[i * S_mres.S_sz + j] = res[0] + res[1] + res[2] + res[3]; //Adds the 4 numbers in res to get one number for this position in mres
        }
    }
}

//Double
void matmul_simd_double(mat& mres, const mat& m1, const mat& m2) {
    mat m2t{ new double[m2.sz * m2.sz], m2.sz };
    Transpose_Double(m2, mres, m2t);

    
    for (int i = 0; i < mres.sz; i++) { 
        for (int j = 0; j < mres.sz; j++) { 
            __m256d result = _mm256_setzero_pd(); 
            for (int k = 0; k < mres.sz; k += 4) {  
                __m256d m1_values = _mm256_load_pd(&m1.data[i * mres.sz + k]); 
                __m256d m2_values = _mm256_loadu_pd(&m2t.data[j * mres.sz + k]); 

                __m256d product = _mm256_mul_pd(m1_values, m2_values); 

                result = _mm256_add_pd(result, product); 
            }

            double res[4]; 
            _mm256_store_pd(res, result); 
            mres.data[i * mres.sz + j] = res[0] + res[1] + res[2] + res[3]; 
        }
    }
}

#pragma endregion

#pragma region Print
//Single
void S_print_mat(const Smat& sm) {
    for (int i = 0; i < sm.S_sz; i++) {
        for (int j = 0; j < sm.S_sz; j++) {
            std::cout << std::setw(3) << sm.s_data[i * sm.S_sz + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}
//Double
void print_mat(const mat& m) {
    for (int i = 0; i < m.sz; i++) {
        for (int j = 0; j < m.sz; j++) {
            std::cout << std::setw(3) << m.data[i * m.sz + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

#pragma endregion

#pragma region Initialise Matrix
//Single
void S_init_mat(Smat& sm) {
    int count = 1;
    for (int i = 0; i < sm.S_sz; i++) {
        for (int j = 0; j < sm.S_sz; j++) {
            sm.s_data[i * sm.S_sz + j] = count++;
            if (count == 10)
                count = 1;
        }
    }
}

//Double
void init_mat(mat& m) {
    int count = 1;
    for (int i = 0; i < m.sz; i++) {
        for (int j = 0; j < m.sz; j++) {
            m.data[i * m.sz + j] = count++;
            if (count == 1)
                count = 1;
        }
    }
}
#pragma endregion

#pragma region Standard Deviation
//Single
float S_standard_deviation(const std::vector<float>& s_data)
{
    float mean = std::accumulate(s_data.begin(), s_data.end(), 0.0) / s_data.size();
    float sum_sqr_diffs = 0.0;
    for (const auto& x : s_data) {
        sum_sqr_diffs += (x - mean) * (x - mean);
    }
    float variance = sum_sqr_diffs / (s_data.size() - 1);
    return std::sqrt(variance);
}

//Double
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
#pragma endregion


int main(int argc, char *argv[])
{
  std::vector<double> data(6);
  std::vector<float> s_data(6);

  unsigned int SZ = 1 << 3; // (1 << 10) == 1024 (Matrix size is 8)

  mat mres{new double[SZ*SZ],SZ},m{new double[SZ*SZ],SZ},id{new double[SZ*SZ],SZ};
  mat mres_parallel{ new double[SZ * SZ],SZ };
  mat mres_parallel_simd{ new double[SZ * SZ],SZ };
  mat mres_simd{new double[SZ*SZ],SZ};

  Smat Smres{new float[SZ * SZ],SZ }, sm{ new float[SZ * SZ],SZ }, Sid{ new float[SZ * SZ],SZ };
  Smat Smres_parallel{ new float[SZ * SZ],SZ };
  Smat Smres_parallel_simd{ new float[SZ * SZ],SZ };
  Smat Smres_simd{ new float[SZ * SZ],SZ };

  using namespace std::chrono;
  using tp_t = time_point<high_resolution_clock>;
  tp_t t1, t2;

  // Matrix description
  std::cout << "Each " << SZ << 'x' << SZ;
  std::cout << " matrix is " << sizeof(double)*SZ*SZ << " bytes.\n";
  init_mat(m); // Initialise Matrix with Double Precision values 
  S_init_mat(sm);// Initialise Matrix with Single Precision values

 
//Timing Simple Multiplication
#pragma region Simple multiplication output 
  std::cout << " " << std::endl;
  std::cout << "Simple multiplication:" << std::endl;
#pragma region Single
  for (int i = 0; i <= 5; i++) {
      t1 = high_resolution_clock::now();
      S_matmul(Smres, sm, sm); 
      t2 = high_resolution_clock::now();

      float d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "single precision timing " << i + 1 << ": " << d << "\n";
      s_data[i] = d;
  }
  std::cout << "single precision average: " << (s_data[0] + s_data[1] + s_data[2] + s_data[3] + s_data[4] + s_data[5]) / 6 << "\n";
  std::cout << "single precision standard deviation: " << S_standard_deviation(s_data) << "\n";
  std::cout << " " << std::endl;
#pragma endregion
#pragma region Double
  for (int i = 0; i <= 5; i++) {
      t1 = high_resolution_clock::now();
      matmul(mres, m, m); 
      t2 = high_resolution_clock::now();

      double d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "double precision timing " << i + 1 << ": " << d << "\n";
      data[i] = d;
  }
  std::cout << "double precision average: " << (data[0] + data[1] + data[2] + data[3] + data[4] + data[5]) / 6 << "\n";
  std::cout << "double precision standard deviation: " << standard_deviation(data) << "\n";
#pragma endregion
#pragma endregion
// Timing SIMD
#pragma region SIMD output
  std::cout << " " << std::endl;
  std::cout << "Simd:" << std::endl;
#pragma region Single
  for (int i = 0; i <= 5; i++) {

      t1 = high_resolution_clock::now();
      S_matmul_simd(Smres_simd, sm, sm); 
      t2 = high_resolution_clock::now();

      float d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "single precision timing " << i + 1 << ": " << d << "\n";
      s_data[i] = d;
  }
  std::cout << "single precision average: " << (s_data[0] + s_data[1] + s_data[2] + s_data[3] + s_data[4] + s_data[5]) / 6 << "\n";
  std::cout << "Simd single precision standard deviation: " << S_standard_deviation(s_data) << "\n";
  std::cout << " " << std::endl;
#pragma endregion
#pragma region Double
  for (int i = 0; i <= 5; i++) {

      t1 = high_resolution_clock::now();
      matmul_simd_double(mres_simd, m, m); 
      t2 = high_resolution_clock::now();

      double d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "double precision timing " << i + 1 << ": " << d << "\n";
      data[i] = d;
  }
  std::cout << "double precision average: " << (data[0] + data[1] + data[2] + data[3] + data[4] + data[5]) / 6 << "\n";
  std::cout << "Simd double precision standard deviation: " << standard_deviation(data) << "\n";
#pragma endregion
#pragma endregion
//Timing Parallel Multiplication
#pragma region Parrallel output
  std::cout << " " << std::endl;
  std::cout << "Parallel:" << std::endl;
  int num_threads = 4; // Num of threads
#pragma region Single
  for (int i = 0; i <= 5; i++) {
      t1 = high_resolution_clock::now();
      S_matmul_parallel(Smres_parallel, sm, sm, num_threads);
      t2 = high_resolution_clock::now();

      float d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "single precision timing " << i + 1 << ": " << d << "\n";
      s_data[i] = d;
  }
  std::cout << "single precision average: " << (s_data[0] + s_data[1] + s_data[2] + s_data[3] + s_data[4] + s_data[5]) / 6 << "\n";
  std::cout << "Parallel single precision standard deviation: " << S_standard_deviation(s_data) << "\n";
  std::cout << " " << std::endl;
#pragma endregion
#pragma region Double 
  for (int i = 0; i <= 5; i++) {
      t1 = high_resolution_clock::now();
      matmul_parallel(mres_parallel, m, m, num_threads);
      t2 = high_resolution_clock::now();

      double d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "double precision timing " << i + 1 << ": " << d << "\n";
      data[i] = d;
  }
  std::cout << "double precision average: " << (data[0] + data[1] + data[2] + data[3] + data[4] + data[5]) / 6 << "\n";
  std::cout << "Parallel double precision standard deviation: " << standard_deviation(data) << "\n";

#pragma endregion
#pragma endregion
// Timing Parallel Multiplication with SIMD
#pragma region Parallel with SIMD output
  std::cout << " " << std::endl;
  std::cout << "Parallel with Simd:" << std::endl;
#pragma region Single
  for (int i = 0; i <= 5; i++) {
      t1 = high_resolution_clock::now();
      matmul_parallel_simd(Smres_parallel_simd, sm, sm, num_threads); 
      t2 = high_resolution_clock::now();

      float d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "single precision timing " << i + 1 << ": " << d << "\n";
      s_data[i] = d;
  }
  std::cout << "single precision average: " << (s_data[0] + s_data[1] + s_data[2] + s_data[3] + s_data[4] + s_data[5]) / 6 << "\n";
  std::cout << "Parallel simd single precision standard deviation: " << S_standard_deviation(s_data) << "\n";
  std::cout << " " << std::endl;
#pragma endregion
#pragma region Double
  for (int i = 0; i <= 5; i++) {
      t1 = high_resolution_clock::now();
      matmul_parallel_simd_double(mres_parallel_simd, m, m, num_threads); 
      t2 = high_resolution_clock::now();

      double d = duration_cast<microseconds>(t2 - t1).count();
      std::cout << "double precision timing " << i + 1 << ": " << d << "\n";
      data[i] = d;
  }
  std::cout << "double precision average: " << (data[0] + data[1] + data[2] + data[3] + data[4] + data[5]) / 6 << "\n";
  std::cout << "Parallel simd double precision standard deviation: " << standard_deviation(data) << "\n";

#pragma endregion
#pragma endregion

#pragma region Asserts
  // Assert statement to check if both multiplications are the same.
  bool b1 = mres == mres_parallel;
  bool b2 = mres == mres_simd;
  bool b3 = mres == mres_parallel;
  bool b4 = mres == mres_parallel_simd;
  bool b5 = Smres == Smres_parallel;
  bool b6 = Smres == Smres_simd;
  bool b7 = Smres == Smres_parallel;
  bool b8 = Smres == Smres_parallel_simd;
  assert(b1);
  assert(b2);
  assert(b3); 
  assert(b4);
  assert(b5);
  assert(b6);
  assert(b7);
  assert(b8);
#pragma endregion

#pragma region Cleanup
  delete[] mres.data;
  delete[] mres_parallel.data;
  delete[] mres_parallel_simd.data;
  delete[] mres_simd.data;
  delete[] m.data;
  delete[] id.data;
  delete[] Smres.s_data;
  delete[] Smres_parallel.s_data;
  delete[] Smres_parallel_simd.s_data;
  delete[] Smres_simd.s_data;
  delete[] sm.s_data;
  delete[] Sid.s_data;
#pragma endregion


  return  0;
}
