#include <iostream>
#include <vector>

using namespace std;

double quadratic(const vector<double> &X);

vector<double> quadratic_grad(const vector<double> &X);

double rosenbrock(const vector<double> &X);

vector<double> rosenbrock_grad(const vector<double> &X);