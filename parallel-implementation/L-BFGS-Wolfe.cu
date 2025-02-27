#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <functional>

#include <cmath>
#include <chrono>
#include <algorithm> 
#include <random> 

using namespace std;


// Armijo and Wolfe condition constants
constexpr double C1 = 1e-4;   // Armijo condition constant
constexpr double C2 = 0.7;    // Wolfe condition constant

// Initial step size for line search
constexpr double INITIAL_STEP_SIZE = 1.0;

// Backtracking parameters
constexpr double BACKTRACKING_ALPHA = 0.5;  // Step reduction factor
constexpr double BACKTRACKING_TOL = 1e-8;  // Convergence threshold

// Wolfe interpolation parameters
constexpr double WOLFE_INTERP_MIN = 1e-10;
constexpr double WOLFE_INTERP_MAX = 10.0;


constexpr double MAX_STEP_SIZE = 10.0;
constexpr double MIN_STEP_SIZE = 1e-6;



// CUDA kernel for updating s and y
// CUDA kernel for sk ← xk+1 − xk. && CUDA kernel for yk ← gk+1 − gk.
__global__ void updateVectors(
    double* s, const double* x_new, const double* x_old,
    double* y, const double* g_new, const double* g_old,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        s[idx] = x_new[idx] - x_old[idx];
        y[idx] = g_new[idx] - g_old[idx];
    }
}


__global__ void copyGradient(double* q, const double* g, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        q[idx] = g[idx];
    }
}


// CUDA kernel for dk ← −r.
__global__ void negateVector(const double* r, double* d, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        d[idx] = -r[idx];
    }
}

// CUDA kernel for xk+1 ← xk + αkdk.
__global__ void updateSolution(const double* x_k, const double* d_k, double* x_next, double alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        x_next[idx] = x_k[idx] + alpha * d_k[idx];
    }
}

__global__ void scaleByRho(const double* input, double* output, double rho, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] * rho;
    }
}






// Utility functions for error handling
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << msg << " (" << cudaGetErrorString(err) << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t err, const char* msg) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        cerr << msg << " (" << (err == CUBLAS_STATUS_NOT_INITIALIZED ? "Not initialized" : "Other error") << ")" << endl;
        exit(EXIT_FAILURE);
    }
}



// Helper functions from sequential code
void ensureSameSize(const vector<double> &v1, const vector<double> &v2) {
    if (v1.size() != v2.size())
        throw logic_error("Vectors must be of same size");
}

double dotProduct(const vector<double> &v1, const vector<double> &v2) {
    ensureSameSize(v1, v2);

    double sum = 0.;
    for (int i = 0; i < v1.size(); ++i) 
        sum += v1[i] * v2[i];

    return sum;
}

vector<double> scalarProduct(const double scalar, const vector<double> &v) {
    vector<double> result = v;

    for (double &element : result)
        element *= scalar;

    return result;
}

vector<double> add(const vector<double> &v1, const vector<double> &v2) {
    ensureSameSize(v1, v2);

    vector<double> sum(v1.size());

    for (int i = 0; i < v1.size(); ++i)
        sum[i] = v1[i] + v2[i];

    return sum;
}



// Helper functions for interpolation
double cubicInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1, double dphi1) {
    double d1 = dphi0 + dphi1 - 3*(phi1 - phi0)/(alpha1 - alpha0);
    double d2 = std::copysign(sqrt(d1*d1 - dphi0*dphi1), alpha1 - alpha0);
    return alpha0 + (alpha1 - alpha0)*(dphi0 + d2 - d1)/(dphi0 - dphi1 + 2*d2);
}

double quadraticInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1) {
    return alpha0 - 0.5*dphi0*alpha0*alpha0/(phi1 - phi0 - dphi0*alpha0);
}



// Safe cubic interpolation with bounds and numerical checks
double safeCubicInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1, double dphi1) {
    if (alpha0 > alpha1) {
        std::swap(alpha0, alpha1);
        std::swap(phi0, phi1);
        std::swap(dphi0, dphi1);
    }
    
    double d1;
    try {
        d1 = dphi0 + dphi1 - 3*(phi1 - phi0)/(alpha1 - alpha0);
    } catch (...) {
        return 0.5 * (alpha0 + alpha1);  
    }
    
    if (std::isnan(d1) || std::isinf(d1)) {
        return 0.5 * (alpha0 + alpha1);
    }
    
    double discriminant = d1*d1 - dphi0*dphi1;
    
    if (discriminant < 0) {
        return 0.5 * (alpha0 + alpha1);
    }
    
    double d2;
    try {
        d2 = std::copysign(sqrt(discriminant), alpha1 - alpha0);
    } catch (...) {
        return 0.5 * (alpha0 + alpha1);  
    }
    
    double denominator = dphi0 - dphi1 + 2*d2;
    
    if (std::abs(denominator) < 1e-10) {
        return 0.5 * (alpha0 + alpha1);
    }
    
    double result;
    try {
        result = alpha0 + (alpha1 - alpha0)*(dphi0 + d2 - d1)/denominator;
    } catch (...) {
        return 0.5 * (alpha0 + alpha1);  
    }
    
    if (std::isnan(result) || std::isinf(result)) {
        return 0.5 * (alpha0 + alpha1);
    }
    
    return std::max(alpha0 + 0.1 * (alpha1 - alpha0), 
                   std::min(alpha1 - 0.1 * (alpha1 - alpha0), result));
}




class CudaStream {
    private:
        cudaStream_t stream_;
        
    public:
        CudaStream() { cudaStreamCreate(&stream_); }
        ~CudaStream() { cudaStreamDestroy(stream_); }
        cudaStream_t get() { return stream_; }
    };



vector<double> LBFGS_CUDA(
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const int max_iterations,
    const int m,
    const double tolerance
) {
    const int size = x0.size();
    cout << "Starting" << endl;

    CudaStream compute_stream;
    CudaStream transfer_stream;
    CudaStream line_search_stream;  

    double *d_x, *d_g, *d_d, *d_q, *d_r;
    checkCudaError(cudaMalloc(&d_x, size * sizeof(double)), "Failed to allocate memory for d_x");
    checkCudaError(cudaMalloc(&d_g, size * sizeof(double)), "Failed to allocate memory for d_g");
    checkCudaError(cudaMalloc(&d_d, size * sizeof(double)), "Failed to allocate memory for d_d");
    checkCudaError(cudaMalloc(&d_q, size * sizeof(double)), "Failed to allocate memory for d_q");
    checkCudaError(cudaMalloc(&d_r, size * sizeof(double)), "Failed to allocate memory for d_r");

    double *d_x_new, *d_g_new, *d_x_temp;
    checkCudaError(cudaMalloc(&d_x_new, size * sizeof(double)), "Failed to allocate memory for d_x_new");
    checkCudaError(cudaMalloc(&d_g_new, size * sizeof(double)), "Failed to allocate memory for d_g_new");
    checkCudaError(cudaMalloc(&d_x_temp, size * sizeof(double)), "Failed to allocate memory for d_x_temp");

    double *h_f_x, *h_f_new, *h_grad_dot_d, *h_dphi_new;
    checkCudaError(cudaMallocHost(&h_f_x, sizeof(double)), "Failed to allocate pinned memory for h_f_x");
    checkCudaError(cudaMallocHost(&h_f_new, sizeof(double)), "Failed to allocate pinned memory for h_f_new");
    checkCudaError(cudaMallocHost(&h_grad_dot_d, sizeof(double)), "Failed to allocate pinned memory for h_grad_dot_d");
    checkCudaError(cudaMallocHost(&h_dphi_new, sizeof(double)), "Failed to allocate pinned memory for h_dphi_new");

    std::vector<double*> s_history(m), y_history(m);
    for (int i = 0; i < m; ++i) {
        checkCudaError(cudaMalloc(&s_history[i], size * sizeof(double)), "Failed to allocate memory for s_history");
        checkCudaError(cudaMalloc(&y_history[i], size * sizeof(double)), "Failed to allocate memory for y_history");
    }

    std::vector<double> gradient(size, 0.0);

    checkCudaError(cudaMemcpyAsync(d_x, x0.data(), size * sizeof(double), 
                                    cudaMemcpyHostToDevice, transfer_stream.get()), 
                    "Failed to copy x0 to d_x asynchronously");

    cublasHandle_t cublasHandle;
    checkCublasError(cublasCreate(&cublasHandle), "Failed to initialize cuBLAS");

    checkCublasError(cublasSetStream(cublasHandle, compute_stream.get()), 
                    "Failed to set compute stream for cuBLAS");

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    vector<double> alpha(m);
    vector<double> rho(m);

    checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                    "Failed to synchronize transfer stream after initial data copy");

    std::vector<double> x_host(size);
    checkCudaError(cudaMemcpyAsync(x_host.data(), d_x, size * sizeof(double), 
                                    cudaMemcpyDeviceToHost, transfer_stream.get()),
                    "Failed to copy d_x to host for initial evaluation");
    checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                    "Failed to synchronize transfer stream");
    
    double initial_f = f(x_host);
    gradient = grad(x_host);

    checkCudaError(cudaMemcpyAsync(d_g, gradient.data(), size * sizeof(double), 
                                    cudaMemcpyHostToDevice, transfer_stream.get()), 
                    "Failed to copy gradient to d_g asynchronously");
    checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                    "Failed to synchronize transfer stream after gradient copy");

    for (int k = 0; k < max_iterations; ++k) {
        if (k == 0) {
            negateVector<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(d_g, d_d, size);
        } else {
            checkCudaError(cudaMemcpyAsync(d_q, d_g, size * sizeof(double), 
                                            cudaMemcpyDeviceToDevice, compute_stream.get()), 
                            "Failed to copy gradient to q asynchronously");

            for (int i = k - 1; i >= max(0, k - m); --i) {
                double si_yi;
                checkCublasError(cublasDdot(cublasHandle, size, s_history[i % m], 1, y_history[i % m], 1, &si_yi),
                                "CUBLAS ddot for si_yi failed");
                
                if (si_yi <= 1e-10) continue;  
                
                rho[i % m] = 1.0 / si_yi;
                
                double si_q;
                checkCublasError(cublasDdot(cublasHandle, size, s_history[i % m], 1, d_q, 1, &si_q),
                                "CUBLAS ddot for si_q failed");
                
                alpha[i % m] = rho[i % m] * si_q;
                double neg_alpha = -alpha[i % m];
                checkCublasError(cublasDaxpy(cublasHandle, size, &neg_alpha, y_history[i % m], 1, d_q, 1),
                                "CUBLAS daxpy failed");
            }

            if (k > 0) {
                double ys, yy;
                int last = (k - 1) % m;
                checkCublasError(cublasDdot(cublasHandle, size, s_history[last], 1, y_history[last], 1, &ys),
                                "CUBLAS ddot for ys failed");
                checkCublasError(cublasDdot(cublasHandle, size, y_history[last], 1, y_history[last], 1, &yy),
                                "CUBLAS ddot for yy failed");
                
                if (yy > 0 && ys > 1e-10) {  
                    double gamma = ys / yy;
                    scaleByRho<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(d_q, d_r, gamma, size);
                } else {
                    double gamma = 1.0;
                    scaleByRho<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(d_q, d_r, gamma, size);
                }
            } else {
                checkCudaError(cudaMemcpyAsync(d_r, d_q, size * sizeof(double), 
                                                cudaMemcpyDeviceToDevice, compute_stream.get()),
                                "Failed to copy q to r asynchronously");
            }

            for (int i = max(0, k - m); i < k; ++i) {
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

        double grad_dot_d;
        checkCublasError(cublasDdot(cublasHandle, size, d_g, 1, d_d, 1, &grad_dot_d),
                        "CUBLAS ddot for grad_dot_d failed");

        double alpha_current = INITIAL_STEP_SIZE;
        double alpha_lo = 0.0;
        double alpha_hi = numeric_limits<double>::infinity();
        double f_lo = initial_f;
        double dphi_lo = grad_dot_d;

        *h_f_x = f(x_host);
        *h_grad_dot_d = grad_dot_d;

        bool line_search_success = false;
        for (int iter = 0; iter < 20; ++iter) {
            updateSolution<<<numBlocks, threadsPerBlock, 0, line_search_stream.get()>>>(
                d_x, d_d, d_x_temp, alpha_current, size);
            
            checkCudaError(cudaStreamSynchronize(line_search_stream.get()), 
                            "Failed to synchronize line search stream");
            
            checkCudaError(cudaMemcpyAsync(x_host.data(), d_x_temp, size * sizeof(double), 
                                            cudaMemcpyDeviceToHost, transfer_stream.get()),
                            "Failed to copy d_x_temp to host");
            checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                            "Failed to synchronize transfer stream");

            *h_f_new = f(x_host);

            if (*h_f_new > *h_f_x + C1 * alpha_current * *h_grad_dot_d || 
                (*h_f_new >= f_lo && iter > 0)) {
                alpha_hi = alpha_current;
                alpha_current = safeCubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, *h_f_new, 
                                                    (*h_f_new - *h_f_x - *h_grad_dot_d * alpha_current)/
                                                    (alpha_current * alpha_current));
                continue;
            }

            gradient = grad(x_host);

            checkCudaError(cudaMemcpyAsync(d_g_new, gradient.data(), size * sizeof(double), 
                                            cudaMemcpyHostToDevice, transfer_stream.get()),
                            "Failed to copy new gradient to d_g_new");
            checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                            "Failed to synchronize transfer stream");

            double dphi_new;
            checkCublasError(cublasDdot(cublasHandle, size, d_g_new, 1, d_d, 1, &dphi_new),
                            "CUBLAS ddot for dphi_new failed");
            *h_dphi_new = dphi_new;

            if (fabs(*h_dphi_new) <= -C2 * *h_grad_dot_d) {
                line_search_success = true;
                break;
            }
            
            if (*h_dphi_new >= 0) {
                alpha_hi = alpha_current;
                alpha_current = safeCubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, *h_f_new, *h_dphi_new);
            } else {
                alpha_lo = alpha_current;
                f_lo = *h_f_new;
                dphi_lo = *h_dphi_new;
                
                if (alpha_hi == numeric_limits<double>::infinity()) {
                    alpha_current *= 2;
                } else {
                    alpha_current = safeCubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, *h_f_new, *h_dphi_new);
                }
            }
            
            if (alpha_current < WOLFE_INTERP_MIN) {
                alpha_current = WOLFE_INTERP_MIN;
                updateSolution<<<numBlocks, threadsPerBlock, 0, line_search_stream.get()>>>(
                    d_x, d_d, d_x_temp, alpha_current, size);
                checkCudaError(cudaStreamSynchronize(line_search_stream.get()), 
                                "Failed to synchronize line search stream");
                break;
            }
        }

        cout << "alpha: " << alpha_current << endl;
        
        if (!line_search_success && alpha_current < 1e-10) {
            cout << "Warning: Line search failed at iteration " << k << endl;
            std::vector<double> x_final(size);
            checkCudaError(cudaMemcpyAsync(x_final.data(), d_x, size * sizeof(double), 
                                            cudaMemcpyDeviceToHost, transfer_stream.get()),
                            "Failed to copy d_x to host asynchronously");
            checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                            "Failed to synchronize transfer stream");

            cudaFreeHost(h_f_x);
            cudaFreeHost(h_f_new);
            cudaFreeHost(h_grad_dot_d);
            cudaFreeHost(h_dphi_new);
            
            return x_final;
        }

        updateSolution<<<numBlocks, threadsPerBlock, 0, compute_stream.get()>>>(
            d_x, d_d, d_x_new, alpha_current, size);

        checkCudaError(cudaStreamSynchronize(compute_stream.get()), 
                        "Failed to synchronize compute stream after solution update");

        std::vector<double> x_new_host(size);
        checkCudaError(cudaMemcpyAsync(x_new_host.data(), d_x_new, size * sizeof(double), 
                                        cudaMemcpyDeviceToHost, transfer_stream.get()),
                        "Failed to copy d_x_new to host asynchronously");
        
        checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                        "Failed to synchronize transfer stream");

        if (!line_search_success) {
            std::vector<double> g_new_host = grad(x_new_host);

            checkCudaError(cudaMemcpyAsync(d_g_new, g_new_host.data(), size * sizeof(double), 
                                            cudaMemcpyHostToDevice, transfer_stream.get()),
                            "Failed to copy updated gradient to d_g_new asynchronously");

            checkCudaError(cudaStreamSynchronize(transfer_stream.get()), 
                            "Failed to synchronize transfer stream after gradient update");
        }

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
        
        if (norm_g <= tolerance) {
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
    cudaFree(d_x_new);
    cudaFree(d_g_new);
    cudaFree(d_x_temp);

    cudaFreeHost(h_f_x);
    cudaFreeHost(h_f_new);
    cudaFreeHost(h_grad_dot_d);
    cudaFreeHost(h_dphi_new);
    
    for (int i = 0; i < m; ++i) {
        cudaFree(s_history[i]);
        cudaFree(y_history[i]);
    }
    cublasDestroy(cublasHandle);
    
    return x_final;
}



double quadratic(const vector<double> &X) {
    double sum = 0.0;
    for (const double x : X) {
        sum += (x - 1) * (x - 1);
    }
    return sum;
}

vector<double> quadratic_grad(const vector<double> &X) {
    vector<double> grad(X.size());
    for (size_t i = 0; i < X.size(); i++) {
        grad[i] = 2.0 * (X[i] - 1);
    }
    return grad;
}

double rosenbrock(const vector<double> &X) {
    double sum = 0.0;
    for (size_t i = 0; i < X.size()-1; i++) {
        double term1 = X[i+1] - X[i] * X[i];
        double term2 = 1 - X[i];
        sum += 100.0 * term1 * term1 + term2 * term2; 
    }
    return sum;
}

vector<double> rosenbrock_grad(const vector<double> &X) {
    vector<double> grad(X.size(), 0.0);
    for (size_t i = 0; i < X.size()-1; i++) {
        double term1 = 2.0 * (X[i] - 1);
        double term2 = X[i+1] - X[i] * X[i];
        grad[i] += term1 - 400.0 * X[i] * term2;
        grad[i+1] += 200.0 * term2;
    }
    return grad;
}

int main() {
    unsigned seed = 42;  
    std::mt19937 gen(seed); 
    std::uniform_real_distribution<> dis(-2, 2); 
    std::vector<double> x0(50000);
    for (double& num : x0) {
        num = dis(gen); 
    }

    cout << "First x: ";
    for (double xi : x0) {
        cout << xi << " ";
    }

    std::vector<double> optimum = LBFGS_CUDA(rosenbrock, rosenbrock_grad, x0, 50000, 10, 1e-1);

    cout << "Found solution: ";
    for (double xi : optimum) {
        cout << xi << " ";
    }
    cout << endl;
    cout << "Optimum value: " << rosenbrock(optimum) << endl;

    return 0;
}
