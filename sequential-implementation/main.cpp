#include <iostream>
#include <math.h>
#include <ctime>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <functional>

using namespace std;

// Utility functions
void printMatrix(const vector<vector<double>> &matrix) {
    for (const vector<double> row : matrix) {
        for (double element : row) 
            cout << element << " ";

        cout << endl;
    }
}

void printVector(const vector<double> &vector) {
    for (const double element : vector)
        cout << element << " ";
    cout << endl;
}

void ensureSameSize(const vector<double> &v1, const vector<double> &v2) {
    if (v1.size() != v2.size())
        throw logic_error("Vectors must be of same size");
}

// ---------------------------------------------------------------------------------------------------------------------
// Functions for testing
double f1(const vector<double> &X) {
    const double x = X[0];
    const double y = X[1];
    const double z = X[2];

    return (x - 1)*(x - 1) + (y + 2)*(y + 2) + (z - 3)*(z - 3);
}

vector<double> grad1(const vector<double> &point) {
    const double x = point[0];
    const double y = point[1];
    const double z = point[2];

    return { 2*x - 2, 2*y + 4, 2*z - 6 };
}

double f2(const vector<double> &X) {
    const double x = X[0];

    return (x * x);
}

vector<double> grad2(const vector<double> &point) {
    const double x = point[0];

    return { 2*x };
}

double f3(const vector<double> &X) {
    const double x = X[0];

    return (x - 2)*(x - 2)*(x - 2)*(x - 2) + 1;
}

vector<double> grad3(const vector<double> &point) {
    const double x = point[0];

    return { 4 * (x - 2)*(x - 2)*(x - 2) };
}

// ---------------------------------------------------------------------------------------------------------------------

double dotProduct(const vector<double> &v1, const vector<double> &v2) {
    ensureSameSize(v1, v2);

    double sum = 0.;
    for (int i = 0; i < v1.size(); ++i) 
        sum += v1[i] * v2[i];

    return sum;
}

vector<double> scalarProduct(const double scalar, const vector<double> &v) {
    vector<double> result = v;

    for (double &element : result)
        element *= scalar;

    return result;
}

vector<double> add(const vector<double> &v1, const vector<double> &v2) {
    ensureSameSize(v1, v2);

    vector<double> sum(v1.size());

    for (int i = 0; i < v1.size(); ++i)
        sum[i] = v1[i] + v2[i];

    return sum;
}

vector<double> negative(const vector<double> &v) {
    vector<double> result;

    for (const double element : v)
        result.push_back(-element);

    return result;
}

/*
    Euclidean norm, usually denoted ||v||.
*/
double vectorNorm(const vector<double> &v) {
    double result = 0.;

    for (const double element : v)
        result += element * element;

    return sqrt(result);
}

double getRho(const vector<double> &s, const vector<double> &y) {
    ensureSameSize(s, y);

    double ys = 0.;
    for (int i = 0; i < s.size(); ++i)
        ys += s[i] * y[i];

    return 1. / ys;
}

double lineSearch(
    const vector<double>& x, 
    const vector<double>& d, 
    const function<double(vector<double>)>& f, 
    const vector<double>& gradient, 
    double alpha = 1.0, 
    double beta = 0.5, 
    double c = 1e-4
) {
    double f_x = f(x);
    vector<double> x_new = add(x, scalarProduct(alpha, d));

    while (f(x_new) > f_x + c * alpha * dotProduct(gradient, d)) {
        alpha *= beta;
        x_new = add(x, scalarProduct(alpha, d));
    }

    return alpha;
}

/*
    Sequential implementation of the LBFGS optimization algorithm.
*/
vector<double> LBFGS(
    const function<double(vector<double>)> f,
    const function<vector<double>(vector<double>)> grad,
    const vector<double> x0,
    const int max_iterations,
    const int m,
    const double beta_min,
    const double beta_max,
    const double tolerance
) {
    const int N = x0.size();

    vector<double> x = x0;
    double f_current = f(x0);
    vector<double> gradient = grad(x);

    vector<vector<double>> s_history, y_history;

    vector<double> d(N, 0);

    for (int k = 0; k < max_iterations; ++k) {

        if (k == 0) {
            d = negative(gradient);
        } else {
            const int bound = max(0, k - m);

            vector<double> s_prev = s_history.back();
            vector<double> y_prev = y_history.back();

            double ys, yy = 0.;

            for (int i = 0; i < N; ++i) {
                ys += s_prev[i] * y_prev[i];
                yy += y_prev[i] * y_prev[i];
            }

            double rho = 1. / ys;
            vector<double> q = gradient;
            vector<double> a(s_history.size(), 0.);

            for (int i = k - 1; i >= bound; --i) {
                vector<double> s_i = s_history[i];
                vector<double> y_i = y_history[i];

                double a_i = 0.;
                for (int j = 0; j < N; ++j)
                    a_i += s_i[j] * q[j];

                a_i = a_i * rho;
                a[i] = a_i;

                for (int j = 0; j < N; ++j)
                    q[j] -= a_i * y_i[j];

                rho = getRho(s_i, y_i);
            }

            vector<double> r = q;
            for (int i = 0; i < N; ++i)
                r[i] *= (ys / yy); // this approximates H

            for (int i = bound; i < k; ++i) {
                vector<double> s_i = s_history[i];
                vector<double> y_i = y_history[i];
                rho = getRho(s_i, y_i);

                double b = 0.;
                for (int j = 0; j < N; ++j)
                    b += y_i[j] * r[j];
                b = b * rho;

                for (int j = 0; j < N; ++j)
                    r[j] += s_i[j] * (a[i] - b);
            }

            d = negative(r);
        }

        double alpha = lineSearch(x, d, f, gradient);
        vector<double> x_new(N);
        for (int i = 0; i < N; ++i)
            x_new[i] = x[i] + alpha * d[i];

        vector<double> gradient_new = grad(x_new);
        f_current = f(x_new);

        vector<double> sk(N), yk(N);
        for (int i = 0; i < N; ++i) {
            sk[i] = x_new[i] - x[i];
            yk[i] = gradient_new[i] - gradient[i];
        }

        s_history.push_back(sk);
        y_history.push_back(yk);

        if (s_history.size() > m) {
            // We only need to keep the last M entries
            s_history.erase(s_history.begin());
            y_history.erase(y_history.begin());
        }

        if (vectorNorm(gradient_new) < tolerance)
            return x_new;

        x = x_new;
        gradient = gradient_new;
    }

    return x;
}


int main() {
    vector<double> optimum1 = LBFGS(f1, grad1, { 0.1, 0.3, 0.4 }, 50, 10, 1e-4, 0.9, 1e-5);
    vector<double> optimum2 = LBFGS(f2, grad2, { 12 }, 50, 10, 1e-4, 0.9, 1e-5);
    vector<double> optimum3 = LBFGS(f3, grad3, { 3. }, 50, 10, 1e-4, 0.9, 1e-5);

    printVector(optimum1); // Solution must be (1 - 2 3)
    printVector(optimum2); // Solution must be 0
    printVector(optimum3); // Solution must be 2

    return 0;
}
