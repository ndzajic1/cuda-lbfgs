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


// CUDA kernel for initializing the Hessian matrix to identity matrix
// CUDA kernel for initialization H0k ← I.
__global__ void initializeHessian(double* H, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size * size) {
        int row = idx / size;
        int col = idx % size;

        // Set diagonal elements to 1 and others to 0
        H[idx] = (row == col) ? 1.0 : 0.0;
    }
}

// CUDA kernel for computing the direction
// CUDA kernel for the product dk ← H0kgk.
__global__ void computeDirection(double* d, const double* H, const double* g, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        double sum = 0.0;
        for (int j = 0; j < size; ++j) {
            sum += H[idx * size + j] * g[j];
        }
        d[idx] = -sum;
    }
}

// CUDA kernel for updating s and y
// this can be split into two kernels if needed
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

// CUDA kernel for dot product of two vectors
/* Each thread computes a partial sum for the coresponding index
 The result is written to shared memory 
 After syncing threads parallel reduction is performed
 Finally, only the first thread in each block adds to result */

// UNNEEDED FOR NOW

// __global__ void vectorDotProduct(const double* vec1, const double* vec2, double* result, int size) {
//     __shared__ double shared_data[256]; // Shared memory for intermediate sums

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int tid = threadIdx.x;

//     double temp_sum = 0.0;

//     if (idx < size) {
//         temp_sum = vec1[idx] * vec2[idx];
//     }
//     shared_data[tid] = temp_sum;

//     __syncthreads();

//     for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
//         if (tid < stride) {
//             shared_data[tid] += shared_data[tid + stride];
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         atomicAdd(result, shared_data[0]);
//     }
// }

//CUDA kernel for H0k ← ys/yy
__global__ void computeHessian(double* H, const double ys, const double yy, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        H[idx] = ys / yy;
    }
}

__global__ void copyGradient(double* q, const double* g, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        q[idx] = g[idx];
    }
}

// CUDA kernel for r ← H0kq
__global__ void hessianVectorProduct(const double* H, const double* q, double* r, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        double temp = 0.0;
        for (int j = 0; j < size; ++j) {
            temp += H[idx * size + j] * q[j];
        }
        r[idx] = temp;
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




// __global__ void vectorScale(const double* input, double* output, double scalar, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < size) {
//         output[idx] = scalar * input[idx];
//     }
// }




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



// pomoćne funkcije za sequential line search, preuzete iz sekvencijalnog koda: 
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


double lineSearch(
    const vector<double>& x, 
    const vector<double>& d, 
    const function<double(vector<double>)>& f, 
    const vector<double>& gradient, 
    double alpha = 1.0, 
    double beta = 0.3, 
    double c = 1e-4
) {
    double f_x = f(x);
    vector<double> x_new = add(x, scalarProduct(alpha, d));
    while (f(x_new) > f_x + c * alpha * dotProduct(gradient, d)) {
        alpha *= beta;
        x_new = add(x, scalarProduct(alpha, d));
    }
    // cout << "alfa " << alpha << endl;
    return alpha;
}






std::vector<double> LBFGS_CUDA(
    const std::function<double(std::vector<double>)>& f,
    const std::function<std::vector<double>(std::vector<double>)>& grad,
    const std::vector<double>& x0,
    int max_iterations,
    int m,
    double beta_min,
    double beta_max,
    double tolerance
) {
    const int size = x0.size();
    

    // Allocate GPU memory
    double *d_x, *d_g, *d_H, *d_d, *d_q, *d_r;
    checkCudaError(cudaMalloc(&d_x, size * sizeof(double)), "Failed to allocate memory for d_x");
    checkCudaError(cudaMalloc(&d_g, size * sizeof(double)), "Failed to allocate memory for d_g");
    checkCudaError(cudaMalloc(&d_H, size * size * sizeof(double)), "Failed to allocate memory for d_H");
    checkCudaError(cudaMalloc(&d_d, size * sizeof(double)), "Failed to allocate memory for d_d");
    checkCudaError(cudaMalloc(&d_q, size * sizeof(double)), "Failed to allocate memory for d_q");
    checkCudaError(cudaMalloc(&d_r, size * sizeof(double)), "Failed to allocate memory for d_r");

    // Histories for s and y (m most recent vectors)
    std::vector<double*> s_history(m), y_history(m);
    for (int i = 0; i < m; ++i) {
        checkCudaError(cudaMalloc(&s_history[i], size * sizeof(double)), "Failed to allocate memory for s_history");
        checkCudaError(cudaMalloc(&y_history[i], size * sizeof(double)), "Failed to allocate memory for y_history");
    }

    std::vector<double> a_history(m, 0.0);  // Host-side storage for a
    std::vector<double> rho_history(m, 0.0);  // Host-side storage for rho
    double* d_a;  // Device storage for a
    double* d_rho;  // Device storage for rho
    checkCudaError(cudaMalloc(&d_a, m * sizeof(double)), "Failed to allocate memory for d_a");
    checkCudaError(cudaMalloc(&d_rho, m * sizeof(double)), "Failed to allocate memory for d_rho");

    // additional allocations
    double* d_x_new;
    checkCudaError(cudaMalloc(&d_x_new, size * sizeof(double)), "Failed to allocate memory for d_x_new");
    std::vector<double> gradient(size, 0.0);  // Initialize gradient to zero or appropriate values
    // gradient still doesn't have correct value so this next line is out of place
    //checkCudaError(cudaMemcpy(d_g, gradient.data(), size * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy gradient to GPU");
    double* d_g_new;
    checkCudaError(cudaMalloc(&d_g_new, size * sizeof(double)), "Failed to allocate memory for d_g_new");



    // Copy initial solution to GPU
    checkCudaError(cudaMemcpy(d_x, x0.data(), size * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy x0 to d_x");

    // Initialize cublas
    cublasHandle_t cublasHandle;
    checkCublasError(cublasCreate(&cublasHandle), "Failed to initialize cuBLAS");

    // Temporary storage for scalars
    double ys, yy, rho;
    double norm_g = 0.0, norm_f = 0.0;

    // Initialization of CUDA params
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    for (int k = 0; k < max_iterations; ++k) {
        if (k == 0) {
            // Initialize Hessian to identity
            numBlocks = (size * size + threadsPerBlock - 1) / threadsPerBlock;
            initializeHessian<<<numBlocks, threadsPerBlock>>>(d_H, size);
            checkCudaError(cudaDeviceSynchronize(), "Kernel initializeHessian failed");


        gradient = grad(x0);
        cout<< "gradient: "<<gradient[0]<<endl;
        // copying gradient from host to device
        checkCudaError(cudaMemcpy(d_g, gradient.data(), size * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy updated gradient to d_g");

            // Compute initial search direction
            numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
            computeDirection<<<numBlocks, threadsPerBlock>>>(d_d, d_H, d_g, size);
            checkCudaError(cudaDeviceSynchronize(), "Kernel computeDirection failed");
            // Allocate host memory to copy d_d
            std::vector<double> d_host(size * size);
            checkCudaError(cudaMemcpy(d_host.data(), d_d, size * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy d_H to host");

            // Output d
            for(double di: d_host){
                cout << " di: "<< di << endl;
            }
        } else {
            // Determine bound
            int bound = max(0, k - m);

            // Compute ys and yy
            double* d_s_prev = s_history[(k - 1) % m]; // cyclic access based on size of m
            double* d_y_prev = y_history[(k - 1) % m];

            checkCublasError(cublasDdot(cublasHandle, size, d_s_prev, 1, d_y_prev, 1, &ys), "CUBLAS ddot for ys failed");
            checkCublasError(cublasDdot(cublasHandle, size, d_y_prev, 1, d_y_prev, 1, &yy), "CUBLAS ddot for yy failed");

            // prevents ys and yy from falling to zero
            // possibly not needed or not allowed, need to check
            cout << "ys: " << ys << ", yy: " << yy << endl;
            if (std::abs(ys) < 1e-8 || std::abs(yy) < 1e-8) {
                std::cerr << "ys or yy is too small. Adjusting values." << std::endl;
                ys = (ys < 0 ? -1 : 1) * std::max(std::abs(ys), 1e-8);
                yy = (yy < 0 ? -1 : 1) * std::max(std::abs(yy), 1e-8);
            }


            // Update H0k
            int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
            // computeHessian<<<numBlocks, threadsPerBlock>>>(d_H, ys, yy, size);
            // checkCudaError(cudaDeviceSynchronize(), "Kernel computeHessian failed");
            computeHessian<<<numBlocks, threadsPerBlock>>>(d_H, ys, yy, size);
            checkCudaError(cudaDeviceSynchronize(), "Kernel computeHessian failed");


            // Pseudocode says to Compute ρk−1 as 1/ys, why ρk−1 instead of ρk
            // Compute ρk−1
            rho = 1.0 / ys;

            // Copy gradient to q
            copyGradient<<<numBlocks, threadsPerBlock>>>(d_q, d_g, size);
            checkCudaError(cudaDeviceSynchronize(), "Kernel copyGradient failed");

            // Backward loop
            for (int i = k - 1; i >= bound; --i) {
                double* d_s_i = s_history[i % m];
                double* d_y_i = y_history[i % m];

                // Compute dot product s_i^T * q
                double ai;
                checkCublasError(cublasDdot(cublasHandle, size, d_s_i, 1, d_q, 1, &ai), "CUBLAS ddot for ai failed");

                // Compute ai = ai * rho[i % m]
                ai *= rho_history[i % m];

                // Store ai for later use
                a_history[i % m] = ai;

                // Update q: q = q - ai * y_i
                // DAXPY: Y←α⋅X+Y
                // in this case: Y = q, α = neg_ai, X = y_i
                double neg_ai = -ai;
                cublasDaxpy(cublasHandle, size, &neg_ai, d_y_i, 1, d_q, 1);
            }


            // Compute r = H0k * q
            hessianVectorProduct<<<numBlocks, threadsPerBlock>>>(d_H, d_q, d_r, size);
            checkCudaError(cudaDeviceSynchronize(), "Kernel hessianVectorProduct failed");

            // Forward loop
            for (int i = bound; i < k; ++i) {
                double* d_s_i = s_history[i % m];
                double* d_y_i = y_history[i % m];

                // Retrieve ai and rho
                double ai = a_history[i % m];

                // Compute dot product y_i^T * r
                double b;
                checkCublasError(cublasDdot(cublasHandle, size, d_y_i, 1, d_r, 1, &b), "CUBLAS ddot for b failed");

                b *= rho_history[i % m];

                // Update r: r = r + s_i * (ai - b)
                double diff = ai - b;
                cublasDaxpy(cublasHandle, size, &diff, d_s_i, 1, d_r, 1);
            }

            // Negate r to get d
            negateVector<<<numBlocks, threadsPerBlock>>>(d_r, d_d, size);
            checkCudaError(cudaDeviceSynchronize(), "Kernel negateVector failed");

            // DEBUGGING FIX: 
            // Normalize the direction vector
            double d_dot_d = 0.0;
            checkCublasError(cublasDdot(cublasHandle, size, d_d, 1, d_d, 1, &d_dot_d), "CUBLAS ddot for d failed");
            double d_norm = std::sqrt(d_dot_d);

            if (d_norm > 1e-8) {  // Avoid division by zero
                double scale = 1.0 / d_norm;
                scaleByRho<<<numBlocks, threadsPerBlock>>>(d_d, d_d, scale, size);
                checkCudaError(cudaDeviceSynchronize(), "Kernel scaleByRho failed for normalizing d");
            }
            // DEBUGGING FIX END
        }

        // Line search
        //double alpha = 1.0; //lineSearchCUDA(d_x, d_d, d_x_new, f, gradient, size);
        // Copy `d_x` and `d_d` from device to host for the sequential line search
        std::vector<double> x_host(size), d_host(size);
        checkCudaError(cudaMemcpy(x_host.data(), d_x, size * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy d_x to host");
        checkCudaError(cudaMemcpy(d_host.data(), d_d, size * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy d_d to host");
        cout << "x_host: " << x_host[0] << ", d_host: " << d_host[0] << endl;

        // Perform sequential line search
        double alpha = lineSearch(x_host, d_host, f, gradient);
        cout << "alpha: " << alpha << endl; 

        // x_new and g_new needed only in this part of code after line search and then they should be stored into d_x and d_g for further evaluation
        // while the memory should be freed

        // Update `d_x_new` on the device using the new alpha value
        updateSolution<<<numBlocks, threadsPerBlock>>>(d_x, d_d, d_x_new, alpha, size);
        checkCudaError(cudaDeviceSynchronize(), "Kernel updateSolution failed");

        // Need to update gradient on host and send it to device before calling the next kernel
        // Step 1: Copy `d_x_new` to host to calculate the new gradient
        std::vector<double> x_new_host(size);
        checkCudaError(cudaMemcpy(x_new_host.data(), d_x_new, size * sizeof(double), cudaMemcpyDeviceToHost), 
                    "Failed to copy d_x_new to host");
        
        // debugging check for x value
        cout << "x_new: ";
        for (double x : x_new_host) cout << x << " ";
        cout << endl;

        // Step 2: Compute the new gradient using the host gradient function
        std::vector<double> g_new_host = grad(x_new_host);

        // Step 3: Copy the updated gradient back to `d_g_new` on the device
        checkCudaError(cudaMemcpy(d_g_new, g_new_host.data(), size * sizeof(double), cudaMemcpyHostToDevice), 
                    "Failed to copy updated gradient to d_g_new");

        // Debugging: Print the updated gradient for verification
        cout << "Updated gradient: ";
        for (double g : g_new_host) {
            cout << g << " ";
        }
        cout << endl;

        //CORRECTED: 
        updateVectors<<<numBlocks, threadsPerBlock>>>(s_history[k % m], d_x_new, d_x, y_history[k % m], d_g_new, d_g, size);
        checkCudaError(cudaDeviceSynchronize(), "Kernel updateVectors failed");

        
        // Getting ready for next iteration
        // Transfer `d_x_new` and `d_g_new` to `d_x` and `d_g`
        checkCudaError(cudaMemcpy(d_x, d_x_new, size * sizeof(double), cudaMemcpyDeviceToDevice), 
                    "Failed to copy d_x_new to d_x");
        checkCudaError(cudaMemcpy(d_g, d_g_new, size * sizeof(double), cudaMemcpyDeviceToDevice), 
                    "Failed to copy d_g_new to d_g");



        // DEBUGGING FIX: 
        // Validate s and y before proceeding
        // if s or y are negative or zero continue with next iteration
        // double s_dot_s = 0.0, y_dot_y = 0.0;
        // checkCublasError(cublasDdot(cublasHandle, size, s_history[k % m], 1, s_history[k % m], 1, &s_dot_s), "CUBLAS ddot for s failed");
        // checkCublasError(cublasDdot(cublasHandle, size, y_history[k % m], 1, y_history[k % m], 1, &y_dot_y), "CUBLAS ddot for y failed");

        // if (s_dot_s < 1e-8 || y_dot_y < 1e-8) {
        //     std::cerr << "Invalid s or y vector. Skipping storage." << std::endl;
        //     continue;
        // }
        // DEBUGGING FIX END



        // OUTPUT CHECK s and y
            std::vector<double> s_host(size), y_host(size);
            cudaMemcpy(s_host.data(), s_history[k % m], size * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(y_host.data(), y_history[k % m], size * sizeof(double), cudaMemcpyDeviceToHost);

            cout << "s: ";
            for (double si : s_host) cout << si << " ";
            cout << endl;

            cout << "y: ";
            for (double yi : y_host) cout << yi << " ";
            cout << endl;
            // OUTPUT CHECK END


        // there is also cublasNorm which combines ddot and square root, can be updated
        // Compute norm of gradient: norm_g = ||g|| using cublasDdot
        double g_dot_g = 0.0;
        checkCublasError(cublasDdot(cublasHandle, size, d_g, 1, d_g, 1, &g_dot_g), "CUBLAS ddot for g failed");
        cout << "g_dot_g:" << g_dot_g << endl;
        norm_g = std::sqrt(g_dot_g);

        // ||f|| = ? how to calculate
        // Compute norm of solution vector: norm_f = ||x|| using cublasDdot
        double x_dot_x = 0.0;
        checkCublasError(cublasDdot(cublasHandle, size, d_x, 1, d_x, 1, &x_dot_x), "CUBLAS ddot for x failed");
        cout << "x_dot_x:" << x_dot_x << endl;
        norm_f = std::sqrt(x_dot_x);


        // Convergence check
        cout << "Iteration " << k << ": norm_g = " << norm_g << ", norm_f = " << norm_f << endl;
        cout << "Optimum value: " << f(x_new_host) << endl;
        if (norm_g / norm_f <= tolerance) {
            cout << "Convergence achieved at iteration " << k << endl;
            break;
        }
    }

    // Copy the final solution back to the host
    std::vector<double> x_final(size);
    checkCudaError(cudaMemcpy(x_final.data(), d_x, size * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy d_x to host");

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_H);
    cudaFree(d_d);
    cudaFree(d_q);
    cudaFree(d_r);
    for (int i = 0; i < m; ++i) {
        cudaFree(s_history[i]);
        cudaFree(y_history[i]);
    }
    cudaFree(d_a);
    cudaFree(d_rho);
    cublasDestroy(cublasHandle);

    return x_final;
}





// int main() {
//     vector<double> x0 = {0.1, 0.3, 0.4};
//     LBFGS_CUDA(nullptr, nullptr, x0, 50, 10, 1e-4, 0.9, 1e-5);
//     return 0;
// }


// double f2(const vector<double> &X) {
//     const double x = X[0];

//     return (x * x);
// }

// vector<double> grad2(const vector<double> &point) {
//     const double x = point[0];

//     return { 2*x };
// }


// int main() {
//     // Initial guess
//     std::vector<double> x0 = {0.1, 0.3, 0.4};

//     // Test function and gradient
//     // auto f = [](const std::vector<double>& x) -> double {
//     //     return (x[0] - 1) * (x[0] - 1) + (x[1] + 2) * (x[1] + 2) + (x[2] - 3) * (x[2] - 3);
//     // };

//     // auto grad = [](const std::vector<double>& x) -> std::vector<double> {
//     //     return {2 * (x[0] - 1), 2 * (x[1] + 2), 2 * (x[2] - 3)};
//     // };

//     // Run the CUDA-based L-BFGS algorithm
//     //std::vector<double> solution = LBFGS_CUDA(f, grad, x0, 50, 10, 1e-4, 0.9, 1e-5);
//     std::vector<double> optimum2 = LBFGS_CUDA(f2, grad2, { 12 }, 10, 10, 1e-4, 0.9, 1e-5);
    
//     // Print the final solution
//     cout << "Found solution: ";
//     for (double xi : optimum2) {
//         cout << xi << " ";
//     }
//     cout << endl;

//     return 0;
// }



// // optimum at: x=3,y=−4.
// double f2D(const std::vector<double> &X) {
//     const double x = X[0];
//     const double y = X[1];

//     // A simple quadratic function: f(x, y) = (x - 3)^2 + (y + 4)^2
//     return (x - 3) * (x - 3) + (y + 4) * (y + 4);
// }

// std::vector<double> grad2D(const std::vector<double> &point) {
//     const double x = point[0];
//     const double y = point[1];

//     // Gradient of f(x, y): grad = [2 * (x - 3), 2 * (y + 4)]
//     return {2 * (x - 3), 2 * (y + 4)};
// }

// int main() {
//     // Initial guess
//     std::vector<double> x0 = {0.0, 0.0};  // Start at (0, 0)

//     // Run the CUDA-based L-BFGS algorithm
//     std::vector<double> optimum2D = LBFGS_CUDA(f2D, grad2D, x0, 100, 100, 1e-4, 0.9, 1e-5);

//     // Print the final solution
//     cout << "Found solution: ";
//     for (double xi : optimum2D) {
//         cout << xi << " ";
//     }
//     cout << endl;

//     return 0;
// }




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
    std::uniform_real_distribution<> dis(-2.048, 2.048); 
    std::vector<double> x0(100);
    for (double& num : x0) {
        num = dis(gen); 
    }

    cout << "First x: ";
    for (double xi : x0) {
        cout << xi << " ";
    }
    // Initial guess
    //std::vector<double> x0 = {0.0, 0.0};  // Start at (0, 0)

    // Run the CUDA-based L-BFGS algorithm
    std::vector<double> optimum = LBFGS_CUDA(rosenbrock, rosenbrock_grad, x0, 5000, 10, 1e-4, 0.9, 1e-5);

    // Print the final solution
    cout << "Found solution: ";
    for (double xi : optimum) {
        cout << xi << " ";
    }
    cout << endl;
    cout << "Optimum value: " << rosenbrock(optimum) << endl;

    return 0;
}
