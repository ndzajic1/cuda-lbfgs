#include <iostream>
#include <vector>

using namespace std;

double quadratic(const vector<double> &X)
{
    double sum = 0.0;
    for (const double x : X)
    {
        sum += (x - 1) * (x - 1);
    }
    return sum;
}

vector<double> quadratic_grad(const vector<double> &X)
{
    vector<double> grad(X.size());
    for (size_t i = 0; i < X.size(); i++)
    {
        grad[i] = 2.0 * (X[i] - 1);
    }
    return grad;
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