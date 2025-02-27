#include <iostream>
#include <vector>

using namespace std;

// Helper functions from sequential code
void ensureSameSize(const vector<double> &v1, const vector<double> &v2)
{
    if (v1.size() != v2.size())
        throw logic_error("Vectors must be of same size");
}

double dotProduct(const vector<double> &v1, const vector<double> &v2)
{
    ensureSameSize(v1, v2);

    double sum = 0.;
    for (int i = 0; i < v1.size(); ++i)
        sum += v1[i] * v2[i];

    return sum;
}

vector<double> scalarProduct(const double scalar, const vector<double> &v)
{
    vector<double> result = v;

    for (double &element : result)
        element *= scalar;

    return result;
}

vector<double> add(const vector<double> &v1, const vector<double> &v2)
{
    ensureSameSize(v1, v2);

    vector<double> sum(v1.size());

    for (int i = 0; i < v1.size(); ++i)
        sum[i] = v1[i] + v2[i];

    return sum;
}