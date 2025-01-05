#ifndef LBFGS_H
#define LBFGS_H


#include<vector>
#include<functional>

using namespace std;

vector<double> LBFGS(
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const int max_iterations,
    const int m,
    const double beta_min,
    const double beta_max,
    const double tolerance
);

void printVector(const vector<double> &vector);


#endif