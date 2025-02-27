#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <functional>

#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include "constants.h"
#include "line_search.h"
#include "functions.h"
#include "vector_utils.h"

using namespace std;

// CUDA kernel for updating s and y
// CUDA kernel for sk ← xk+1 − xk. && CUDA kernel for yk ← gk+1 − gk.
__global__ void updateVectors(
    double *s, const double *x_new, const double *x_old,
    double *y, const double *g_new, const double *g_old,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        s[idx] = x_new[idx] - x_old[idx];
        y[idx] = g_new[idx] - g_old[idx];
    }
}

__global__ void copyGradient(double *q, const double *g, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        q[idx] = g[idx];
    }
}

// CUDA kernel for dk ← −r.
__global__ void negateVector(const double *r, double *d, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        d[idx] = -r[idx];
    }
}

// CUDA kernel for xk+1 ← xk + αkdk.
__global__ void updateSolution(const double *x_k, const double *d_k, double *x_next, double alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        x_next[idx] = x_k[idx] + alpha * d_k[idx];
    }
}

__global__ void scaleByRho(const double *input, double *output, double rho, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = input[idx] * rho;
    }
}

// Utility functions for error handling
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        cerr << msg << " (" << cudaGetErrorString(err) << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t err, const char *msg)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        cerr << msg << " (" << (err == CUBLAS_STATUS_NOT_INITIALIZED ? "Not initialized" : "Other error") << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

class CudaStream
{
private:
    cudaStream_t stream_;

public:
    CudaStream() { cudaStreamCreate(&stream_); }
    ~CudaStream() { cudaStreamDestroy(stream_); }
    cudaStream_t get() { return stream_; }
};

// Kernel to compute x + alpha*d
__global__ void computeTrialPoint(const double *x, const double *d, double *x_trial, double alpha, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        x_trial[idx] = x[idx] + alpha * d[idx];
    }
}

// Kernel to compute dot product of gradient and direction
__global__ void computeDotProduct(const double *gradient, const double *direction, double *result, int n)
{
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? gradient[idx] * direction[idx] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        result[blockIdx.x] = sdata[0];
}

vector<double> LBFGS_CUDA(
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const int max_iterations,
    const int m,
    const double tolerance)
{
    const int size = x0.size();
    cout << "Starting" << endl;

    CudaStream compute_stream;
    CudaStream transfer_stream;

    const double INITIAL_STEP_SIZE = 1.0;
    const double BACKTRACKING_ALPHA = 0.5;
    const double BACKTRACKING_TOL = 1e-10;
    const double C1 = 1e-4;

    double *d_x, *d_g, *d_d, *d_q, *d_r;
    checkCudaError(cudaMalloc(&d_x, size * sizeof(double)), "Failed to allocate memory for d_x");
    checkCudaError(cudaMalloc(&d_g, size * sizeof(double)), "Failed to allocate memory for d_g");
    checkCudaError(cudaMalloc(&d_d, size * sizeof(double)), "Failed to allocate memory for d_d");
    checkCudaError(cudaMalloc(&d_q, size * sizeof(double)), "Failed to allocate memory for d_q");
    checkCudaError(cudaMalloc(&d_r, size * sizeof(double)), "Failed to allocate memory for d_r");

    double *d_x_trial;
    checkCudaError(cudaMalloc(&d_x_trial, size * sizeof(double)), "Failed to allocate memory for d_x_trial");
    double *d_dot_result, *h_dot_result;
    int numBlocks = (size + 255) / 256;
    checkCudaError(cudaMalloc(&d_dot_result, numBlocks * sizeof(double)), "Failed to allocate memory for d_dot_result");
    h_dot_result = new double[numBlocks];

    std::vector<double *> s_history(m), y_history(m);
    for (int i = 0; i < m; ++i)
    {
        checkCudaError(cudaMalloc(&s_history[i], size * sizeof(double)), "Failed to allocate memory for s_history");
        checkCudaError(cudaMalloc(&y_history[i], size * sizeof(double)), "Failed to allocate memory for y_history");
    }

    double *d_x_new;
    double *d_g_new;
    checkCudaError(cudaMalloc(&d_x_new, size * sizeof(double)), "Failed to allocate memory for d_x_new");
    checkCudaError(cudaMalloc(&d_g_new, size * sizeof(double)), "Failed to allocate memory for d_g_new");

    std::vector<double> gradient(size, 0.0);

    checkCudaError(cudaMemcpyAsync(d_x, x0.data(), size * sizeof(double),
                                   cudaMemcpyHostToDevice, transfer_stream.get()),
                   "Failed to copy x0 to d_x asynchronously");

    cublasHandle_t cublasHandle;
    checkCublasError(cublasCreate(&cublasHandle), "Failed to initialize cuBLAS");

    checkCublasError(cublasSetStream(cublasHandle, compute_stream.get()),
                     "Failed to set compute stream for cuBLAS");

    int threadsPerBlock = 256;
    numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    vector<double> alpha(m);
    vector<double> rho(m);

    checkCudaError(cudaStreamSynchronize(transfer_stream.get()),
                   "Failed to synchronize transfer stream after initial data copy");

    for (int k = 0; k < max_iterations; ++k)
    {
        if (k == 0)
        {
            gradient = grad(x0);

            checkCudaError(cudaMemcpyAsync(d_g, gradient.data(), size * sizeof(double),
                                           cudaMemcpyHostToDevice, transfer_stream.get()),
                           "Failed to copy gradient to d_g asynchronously");

            checkCudaError(cudaStreamSynchronize(transfer_stream.get()),
                           "Failed to synchronize transfer stream after gradient copy");

            negateVector<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(d_g, d_d, size);
        }
        else
        {
            checkCudaError(cudaMemcpyAsync(d_q, d_g, size * sizeof(double),
                                           cudaMemcpyDeviceToDevice, compute_stream.get()),
                           "Failed to copy gradient to q asynchronously");

            for (int i = k - 1; i >= max(0, k - m); --i)
            {
                double si_yi;
                checkCublasError(cublasDdot(cublasHandle, size, s_history[i % m], 1, y_history[i % m], 1, &si_yi),
                                 "CUBLAS ddot for si_yi failed");

                if (si_yi <= 1e-10)
                    continue;

                rho[i % m] = 1.0 / si_yi;

                double si_q;
                checkCublasError(cublasDdot(cublasHandle, size, s_history[i % m], 1, d_q, 1, &si_q),
                                 "CUBLAS ddot for si_q failed");

                alpha[i % m] = rho[i % m] * si_q;
                double neg_alpha = -alpha[i % m];
                checkCublasError(cublasDaxpy(cublasHandle, size, &neg_alpha, y_history[i % m], 1, d_q, 1),
                                 "CUBLAS daxpy failed");
            }

            if (k > 0)
            {
                double ys, yy;
                int last = (k - 1) % m;
                checkCublasError(cublasDdot(cublasHandle, size, s_history[last], 1, y_history[last], 1, &ys),
                                 "CUBLAS ddot for ys failed");
                checkCublasError(cublasDdot(cublasHandle, size, y_history[last], 1, y_history[last], 1, &yy),
                                 "CUBLAS ddot for yy failed");

                if (yy > 0 && ys > 1e-10)
                {
                    double gamma = ys / yy;
                    scaleByRho<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(d_q, d_r, gamma, size);
                }
                else
                {
                    double gamma = 1.0;
                    scaleByRho<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(d_q, d_r, gamma, size);
                }
            }
            else
            {
                checkCudaError(cudaMemcpyAsync(d_r, d_q, size * sizeof(double),
                                               cudaMemcpyDeviceToDevice, compute_stream.get()),
                               "Failed to copy q to r asynchronously");
            }

            for (int i = max(0, k - m); i < k; ++i)
            {
                double yi_r;
                checkCublasError(cublasDdot(cublasHandle, size, y_history[i % m], 1, d_r, 1, &yi_r),
                                 "CUBLAS ddot for yi_r failed");

                double beta = rho[i % m] * yi_r;
                double diff = alpha[i % m] - beta;
                checkCublasError(cublasDaxpy(cublasHandle, size, &diff, s_history[i % m], 1, d_r, 1),
                                 "CUBLAS daxpy failed");
            }

            negateVector<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(d_r, d_d, size);
        }

        checkCudaError(cudaStreamSynchronize(compute_stream.get()),
                       "Failed to synchronize compute stream");

        double step_size = INITIAL_STEP_SIZE;
        double directional_derivative;

        computeDotProduct<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(
            d_g, d_d, d_dot_result, size);

        checkCudaError(cudaMemcpy(h_dot_result, d_dot_result, numBlocks * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "Failed to copy dot result to host");

        directional_derivative = 0.0;
        for (int i = 0; i < numBlocks; i++)
        {
            directional_derivative += h_dot_result[i];
        }

        std::vector<double> x_host(size);
        checkCudaError(cudaMemcpy(x_host.data(), d_x, size * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "Failed to copy d_x to host");
        double f_current = f(x_host);

        while (true)
        {
            computeTrialPoint<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(
                d_x, d_d, d_x_trial, step_size, size);

            checkCudaError(cudaStreamSynchronize(compute_stream.get()),
                           "Failed to synchronize compute stream after trial point computation");

            std::vector<double> x_trial_host(size);
            checkCudaError(cudaMemcpy(x_trial_host.data(), d_x_trial, size * sizeof(double),
                                      cudaMemcpyDeviceToHost),
                           "Failed to copy d_x_trial to host");

            double f_trial = f(x_trial_host);

            if (f_trial <= f_current + C1 * step_size * directional_derivative)
            {
                break;
            }

            step_size *= BACKTRACKING_ALPHA;

            if (step_size < BACKTRACKING_TOL)
            {
                step_size = 0.5;
                break;
            }
        }

        cout << "alpha: " << step_size << endl;

        if (step_size < 1e-4)
        {
            cout << "Warning: Line search resulted in very small step size at iteration " << k << endl;
        }

        updateSolution<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(
            d_x, d_d, d_x_new, step_size, size);

        checkCudaError(cudaStreamSynchronize(compute_stream.get()),
                       "Failed to synchronize compute stream after solution update");

        std::vector<double> x_new_host(size);
        checkCudaError(cudaMemcpyAsync(x_new_host.data(), d_x_new, size * sizeof(double),
                                       cudaMemcpyDeviceToHost, transfer_stream.get()),
                       "Failed to copy d_x_new to host asynchronously");

        checkCudaError(cudaStreamSynchronize(transfer_stream.get()),
                       "Failed to synchronize transfer stream");

        std::vector<double> g_new_host = grad(x_new_host);

        checkCudaError(cudaMemcpyAsync(d_g_new, g_new_host.data(), size * sizeof(double),
                                       cudaMemcpyHostToDevice, transfer_stream.get()),
                       "Failed to copy updated gradient to d_g_new asynchronously");

        checkCudaError(cudaStreamSynchronize(transfer_stream.get()),
                       "Failed to synchronize transfer stream after gradient update");

        updateVectors<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(
            s_history[k % m], d_x_new, d_x, y_history[k % m], d_g_new, d_g, size);

        checkCudaError(cudaMemcpyAsync(d_x, d_x_new, size * sizeof(double),
                                       cudaMemcpyDeviceToDevice, compute_stream.get()),
                       "Failed to copy d_x_new to d_x asynchronously");
        checkCudaError(cudaMemcpyAsync(d_g, d_g_new, size * sizeof(double),
                                       cudaMemcpyDeviceToDevice, compute_stream.get()),
                       "Failed to copy d_g_new to d_g asynchronously");

        checkCudaError(cudaStreamSynchronize(compute_stream.get()),
                       "Failed to synchronize compute stream after preparing for next iteration");

        double g_dot_g = 0.0;
        checkCublasError(cublasDdot(cublasHandle, size, d_g, 1, d_g, 1, &g_dot_g),
                         "CUBLAS ddot for g failed");
        double norm_g = std::sqrt(g_dot_g);

        cout << "Iteration " << k << ": norm_g = " << norm_g << endl;
        cout << "Optimum value: " << f(x_new_host) << endl;

        if (norm_g <= tolerance)
        {
            cout << "Convergence achieved at iteration " << k << endl;
            break;
        }
    }

    std::vector<double> x_final(size);
    checkCudaError(cudaMemcpyAsync(x_final.data(), d_x, size * sizeof(double),
                                   cudaMemcpyDeviceToHost, transfer_stream.get()),
                   "Failed to copy d_x to host asynchronously");
    checkCudaError(cudaStreamSynchronize(transfer_stream.get()),
                   "Failed to synchronize transfer stream after final copy");

    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_d);
    cudaFree(d_q);
    cudaFree(d_r);
    cudaFree(d_x_trial);
    cudaFree(d_dot_result);
    delete[] h_dot_result;

    for (int i = 0; i < m; ++i)
    {
        cudaFree(s_history[i]);
        cudaFree(y_history[i]);
    }
    cudaFree(d_x_new);
    cudaFree(d_g_new);
    cublasDestroy(cublasHandle);

    return x_final;
}

int main()
{
    unsigned seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-2, 2);
    std::vector<double> x0(50000);
    for (double &num : x0)
    {
        num = dis(gen);
    }

    cout << "First x: ";
    for (double xi : x0)
    {
        cout << xi << " ";
    }

    std::vector<double> optimum = LBFGS_CUDA(rosenbrock, rosenbrock_grad, x0, 50000, 10, 1e-1);

    cout << "Found solution: ";
    for (double xi : optimum)
    {
        cout << xi << " ";
    }
    cout << endl;
    cout << "Optimum value: " << rosenbrock(optimum) << endl;

    return 0;
}
