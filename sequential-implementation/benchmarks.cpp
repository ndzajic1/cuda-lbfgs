#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <chrono>
#include "lbfgs.h"
#include <algorithm> 
#include <random> 

using namespace std;

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

void benchmark(
    const string function_name,
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const int max_iterations,
    const int m,
    const double beta_min,
    const double beta_max,
    const double tolerance
) {
    auto start = chrono::high_resolution_clock::now();
    
    vector<double> optimum = LBFGS(f, grad, x0, max_iterations, m, beta_min, beta_max, tolerance);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << "Function: " << function_name << endl;
    // cout << "Initial point: ";
    // printVector(x0);
    cout << "Optimum point: ";
    printVector(optimum);
    cout << "Optimum value: " << f(optimum) << endl;
    cout << "Elapsed time: " << elapsed.count() << " seconds" << endl;
    cout << "---------------------------------------------" << endl;
}

int main() {
    unsigned seed = 42;  
    std::mt19937 gen(seed); 
    std::uniform_real_distribution<> dis(-2.048, 2.048); 
    std::vector<double> x0(100);
    for (double& num : x0) {
        num = dis(gen); 
    }


    benchmark(
        "Quadratic Function",
        quadratic,
        quadratic_grad,
        x0, // x0
        100, // max_iterations
        10,  // m
        1e-4, // beta_min
        0.9, // beta_max
        1e-5 // tolerance
    );

    benchmark(
        "Rosenbrock Function",
        rosenbrock,
        rosenbrock_grad,
        x0, // x0
        100, // max_iterations
        10,  // m
        1e-4, // beta_min
        0.9, // beta_max
        1e-5 // tolerance
    );

    return 0;
}
