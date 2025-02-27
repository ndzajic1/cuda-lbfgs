#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

void printMatrix(const vector<vector<double>> &matrix)
{
    for (const vector<double> row : matrix)
    {
        for (double element : row)
            cout << element << " ";

        cout << endl;
    }
}

void printVector(const vector<double> &vector)
{
    for (const double element : vector)
        cout << element << " ";
    cout << endl;
}

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

vector<double> negative(const vector<double> &v)
{
    vector<double> result;

    for (const double element : v)
        result.push_back(-element);

    return result;
}

/*
    Euclidean norm, usually denoted ||v||.
*/
double vectorNorm(const vector<double> &v)
{
    double result = 0.;

    for (const double element : v)
        result += element * element;

    return sqrt(result);
}

double getRho(const vector<double> &s, const vector<double> &y)
{
    ensureSameSize(s, y);

    double ys = 0.;
    for (int i = 0; i < s.size(); ++i)
        ys += s[i] * y[i];

    return 1. / ys;
}

// Function to calculate average of values in a vector
double calculateAverage(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    
    // Using std::accumulate to sum all elements
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    
    // Return average
    return sum / values.size();
}