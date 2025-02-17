__global__ void quadratic_kernel(const double* x, double* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        results[idx] = (x[idx] - 1) * (x[idx] - 1);
    }
}

__global__ void quadratic_grad_kernel(const double* x, double* gradients, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        gradients[idx] = 2.0 * (x[idx] - 1);
    }
}

__global__ void rosenbrock_kernel(const double* x, double* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - 1) {
        double xi = x[idx];
        double x_next = x[idx+1];
        results[idx] = 100.0 * (x_next - xi * xi) * (x_next - xi * xi) + (1 - xi) * (1 - xi);
    }
}

__global__ void rosenbrock_grad_kernel(const double* x, double* gradients, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double grad = 0.0;
        if (idx > 0) {
            double x_prev = x[idx-1];
            grad += -400.0 * x_prev * (x[idx] - x_prev * x_prev);
        }
        if (idx < n - 1) {
            double x_next = x[idx+1];
            grad += 200.0 * (x_next - x[idx] * x[idx]) - 2.0 * (1 - x[idx]);
        }
        gradients[idx] = grad;
    }
}

void quadratic(const double* x, double* results, int n) {
    double *d_x, *d_results;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_results, n * sizeof(double));

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    quadratic_kernel<<<gridSize, blockSize>>>(d_x, d_results, n);
    cudaMemcpy(results, d_results, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_results);
}

void quadratic_grad(const double* x, double* gradients, int n) {
    double *d_x, *d_gradients;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_gradients, n * sizeof(double));

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    quadratic_grad_kernel<<<gridSize, blockSize>>>(d_x, d_gradients, n);
    cudaMemcpy(gradients, d_gradients, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_gradients);
}

void rosenbrock(const double* x, double* results, int n) {
    double *d_x, *d_results;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_results, n * sizeof(double));

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    rosenbrock_kernel<<<gridSize, blockSize>>>(d_x, d_results, n);
    cudaMemcpy(results, d_results, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_results);
}

void rosenbrock_grad(const double* x, double* gradients, int n) {
    double *d_x, *d_gradients;
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_gradients, n * sizeof(double));

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    rosenbrock_grad_kernel<<<gridSize, blockSize>>>(d_x, d_gradients, n);
    cudaMemcpy(gradients, d_gradients, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_gradients);
}