#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <vector_utils.h>
#include <chrono>
#include <lbfgs.h>
#include <cassert>

using namespace std;

// Constant for the coefficient value
const double COEFFICIENT = 1000.0;

// Function that returns a quadratic form function with fixed coefficients
std::function<double(const std::vector<double>&)> generate_quadratic_function(int n) {
    return [n](const std::vector<double>& x) -> double {
        assert(x.size() == n && "Input vector must match specified dimension");
        
        double result = 0.0;
        
        // Diagonal terms (x_i^2 * COEFFICIENT)
        for (int i = 0; i < n; i++) {
            result += COEFFICIENT * x[i] * x[i];
        }
        
        // Cross-terms for i adjacent to j
        for (int i = 0; i < n-1; i++) {
            result += (COEFFICIENT/10.0) * x[i] * x[i+1];
        }
        
        return result;
    };
}

// Function that returns the gradient function of the quadratic form
std::function<std::vector<double>(const std::vector<double>&)> generate_quadratic_gradient(int n) {
    return [n](const std::vector<double>& x) -> std::vector<double> {
        assert(x.size() == n && "Input vector must match specified dimension");
        
        std::vector<double> gradient(n, 0.0);
        
        // Gradient of diagonal terms (2 * COEFFICIENT * x_i)
        for (int i = 0; i < n; i++) {
            gradient[i] = 2.0 * COEFFICIENT * x[i];
        }
        
        // Gradient of cross-terms
        for (int i = 0; i < n-1; i++) {
            gradient[i] += (COEFFICIENT/10.0) * x[i+1];
            gradient[i+1] += (COEFFICIENT/10.0) * x[i];
        }
        
        return gradient;
    };
}

double rosenbrock(const vector<double> &X)
{
    double sum = 0.0;
    for (size_t i = 0; i < X.size() - 1; i++)
    {
        double term1 = X[i + 1] - X[i] * X[i];
        double term2 = 1 - X[i];
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    return sum;
}

vector<double> rosenbrock_grad(const vector<double> &X)
{
    vector<double> grad(X.size(), 0.0);
    for (size_t i = 0; i < X.size() - 1; i++)
    {
        double term1 = 2.0 * (X[i] - 1);
        double term2 = X[i + 1] - X[i] * X[i];
        grad[i] += term1 - 400.0 * X[i] * term2;
        grad[i + 1] += 200.0 * term2;
    }
    return grad;
}

double benchmark(
    const string function_name,
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const int max_iterations,
    const int m,
    const double tolerance
) {

    auto start = chrono::high_resolution_clock::now();
    vector<double> optimum = LBFGS(f, grad, x0, "backtracking", max_iterations, m,tolerance, false);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << "Function: " << function_name << endl;
    cout << "Optimum value: " << f(optimum) << endl;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;
    cout << "---------------------------------------------" << endl;

    return elapsed.count();
}
