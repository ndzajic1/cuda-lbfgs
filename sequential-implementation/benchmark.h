#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <vector_utils.h>
#include <chrono>
#include <lbfgs.h>
#include <matrices.h>

using namespace std;

std::function<double(const std::vector<double>&)> generate_quadratic_function(int n);
std::function<std::vector<double>(const std::vector<double>&)> generate_quadratic_gradient(int n);

double rosenbrock(const vector<double> &X);
vector<double> rosenbrock_grad(const vector<double> &X);

double benchmark(
    const string function_name,
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const int max_iterations,
    const int m,
    const double tolerance);