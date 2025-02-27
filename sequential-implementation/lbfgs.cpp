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
    const bool verbose = true)
{
    const int N = x0.size();
    vector<double> x = x0;
    double f_current = f(x0);
    vector<double> gradient = grad(x);

    deque<vector<double>> s_history, y_history;
    vector<double> d(N);

    // Determine line search function before loop
    function<double(const vector<double> &, const vector<double> &,
                    const function<double(vector<double>)> &, const vector<double> &)>
        line_search;

    if (line_search_method == "backtracking")
    {
        line_search = backtrackingLineSearch;
    }
    else if (line_search_method == "interpolation")
    {
        line_search = armijoInterpolationLineSearch;
    }
    else if (line_search_method == "wolfe")
    {
        // Wrap wolfeInterpolationLineSearch which requires grad
        line_search = [&grad](const vector<double> &x, const vector<double> &d,
                              const function<double(vector<double>)> &f,
                              const vector<double> &gradient)
        {
            return wolfeInterpolationLineSearch(x, d, f, grad, gradient);
        };
    }
    else if (line_search_method == "backtracking_wolfe")
    {
        line_search = [&grad](const vector<double> &x, const vector<double> &d,
                              const function<double(vector<double>)> &f,
                              const vector<double> &gradient)
        {
            return backtrackingWolfeLineSearch(x, d, f, grad, gradient);
        };
    }
    else
    {
        throw invalid_argument("Unknown line search method: " + line_search_method);
    }
    
    for (int k = 0; k < max_iterations; ++k)
    {

        // Print current state for debugging
        if(verbose)
            cout << "Iteration " << k << ", f = " << f_current
                << ", |grad| = " << vectorNorm(gradient) << endl;

        if (vectorNorm(gradient) < tolerance)
        {
            cout << "Converged!" << endl;
            return x;
        }

        // Compute search direction
        if (k == 0 || s_history.empty())
        {
            // Initial Hessian approximation H_0 = I
            d = negative(gradient);
        }
        else
        {
            // Two-loop recursion
            vector<double> q = gradient;
            const int h_size = s_history.size();
            vector<double> alpha(h_size);

            // First loop
            for (int i = h_size - 1; i >= 0; --i)
            {
                double rho = 1.0 / dotProduct(y_history[i], s_history[i]);
                if (!isfinite(rho))
                {
                    cout << "Warning: Invalid rho at iteration " << k << endl;
                    d = negative(gradient);
                    goto perform_line_search; // Skip to line search with gradient direction
                }
                alpha[i] = rho * dotProduct(s_history[i], q);
                for (int j = 0; j < N; ++j)
                {
                    q[j] -= alpha[i] * y_history[i][j];
                }
            }

            // Middle scaling
            double gamma = dotProduct(s_history.back(), y_history.back()) /
                           dotProduct(y_history.back(), y_history.back());
            if (gamma <= 0 || !isfinite(gamma))
            {
                cout << "Warning: Invalid gamma at iteration " << k << endl;
                d = negative(gradient);
                goto perform_line_search;
            }

            vector<double> r = q;
            for (int i = 0; i < N; ++i)
            {
                r[i] *= gamma;
            }

            // Second loop
            for (int i = 0; i < h_size; ++i)
            {
                double rho = 1.0 / dotProduct(y_history[i], s_history[i]);
                double beta = rho * dotProduct(y_history[i], r);
                for (int j = 0; j < N; ++j)
                {
                    r[j] += s_history[i][j] * (alpha[i] - beta);
                }
            }

            d = negative(r);
        }
        
    perform_line_search:
        double grad_dot_d = dotProduct(gradient, d);
        if (grad_dot_d >= 0)
        {
            cout << "Warning: Not a descent direction, using gradient" << endl;
            d = negative(gradient);
            grad_dot_d = dotProduct(gradient, d);
        }

        // Perform backtracking line search using the provided function
        double alpha = line_search(x, d, f, gradient);

        // Compute the new point and its function value
        vector<double> x_new = add(x, scalarProduct(alpha, d));
        double f_new = f(x_new);
        f_current = f_new;

        // Check if line search failed to find a suitable step size
        if (alpha < 1e-10)
        {
            cout << "Warning: Line search failed at iteration " << k << endl;
            return x; // Return current point if line search fails
        }

        // Compute new gradient
        vector<double> gradient_new = grad(x_new);

        // Update histories
        vector<double> s_k(N), y_k(N);
        for (int i = 0; i < N; ++i)
        {
            s_k[i] = x_new[i] - x[i];
            y_k[i] = gradient_new[i] - gradient[i];
        }

        double sy = dotProduct(s_k, y_k);
        if (sy > 0)
        { // Only update if curvature condition is satisfied
            if (s_history.size() >= m)
            {
                s_history.pop_front();
                y_history.pop_front();
            }
            s_history.push_back(s_k);
            y_history.push_back(y_k);
        }
        else
        {
            cout << "Warning: Skipping update, sy = " << sy << endl;
        }

        x = x_new;
        gradient = gradient_new;
    }

    cout << "Maximum iterations reached" << endl;
    return x;
}
