#include <iostream>
#include <ctime>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <functional>
#include <deque>
#include <chrono>
#include <cmath>
#include <random>
#include <vector_utils.h>
#include <config.h>
#include <line_search.h>

using namespace std;

vector<double> LBFGS(
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const string line_search_method,
    const int max_iterations = 1000,
    const int m = 10,
    const double tolerance = 1e-5,
    bool verbose = false);