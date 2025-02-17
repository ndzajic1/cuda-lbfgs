#ifndef LBFGS_CONFIG_H
#define LBFGS_CONFIG_H

#include <vector>
#include <functional>
#include <stdexcept>

using namespace std;

// Enum for different line search methods
enum class LineSearchMethod {
    Backtracking,
    BacktrackingWolfe,
    ArmijoInterpolation,
    WolfeInterpolation
};

// Function pointer types for different line search strategies
using LineSearchFunction = function<double(
    const vector<double>&, 
    const vector<double>&, 
    const function<double(vector<double>)>&, 
    const vector<double>&)>;

using LineSearchFunctionWithGradient = function<double(
    const vector<double>&, 
    const vector<double>&, 
    const function<double(vector<double>)>&, 
    const function<vector<double>(vector<double>)>&, 
    const vector<double>&)>;

// Function to select a line search function based on the chosen method
LineSearchFunction selectLineSearch(LineSearchMethod method);
LineSearchFunctionWithGradient selectLineSearchWithGradient(LineSearchMethod method);

#endif // LBFGS_CONFIG_H
