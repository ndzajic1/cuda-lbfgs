#ifndef LBFGS_H
#define LBFGS_H

#include <vector>
#include <functional>
#include <LineSearch.h>
#include <LBFGSConfig.h>
#include <LBFGSConstants.h>
#include <VectorOps.h>
#include "matrices.h"

using namespace std;

vector<double> LBFGS(
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const string line_search_method,
    const int max_iterations,
    const int m,
    const double tolerance
);

#endif
