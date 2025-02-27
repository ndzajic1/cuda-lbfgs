#include <iostream>
#include "constants.h"
#include <vector>
#include "vector_utils.h"
#include <functional>
#include <limits>
#include <cmath>

using namespace std;

// Helper functions for interpolation
double cubicInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1, double dphi1)
{
    double d1 = dphi0 + dphi1 - 3 * (phi1 - phi0) / (alpha1 - alpha0);
    double d2 = std::copysign(sqrt(d1 * d1 - dphi0 * dphi1), alpha1 - alpha0);
    return alpha0 + (alpha1 - alpha0) * (dphi0 + d2 - d1) / (dphi0 - dphi1 + 2 * d2);
}

double quadraticInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1)
{
    return alpha0 - 0.5 * dphi0 * alpha0 * alpha0 / (phi1 - phi0 - dphi0 * alpha0);
}

// Backtracking line search (Armijo condition)
double backtrackingLineSearch(const vector<double> &x, const vector<double> &d,
                              const function<double(vector<double>)> &f,
                              const vector<double> &gradient)
{
    double alpha = INITIAL_STEP_SIZE;

    while (f(x) - f(add(x, scalarProduct(alpha, d))) < C1 * alpha * dotProduct(gradient, d))
    {
        alpha *= BACKTRACKING_ALPHA;
        if (alpha < BACKTRACKING_TOL)
            break;
    }

    if (alpha < 1e-4)
    {
        return 0.5; // Prevent excessively small steps
    }
    return alpha;
}

double backtrackingWolfeLineSearch(
    const std::vector<double> &x,
    const std::vector<double> &d,
    const std::function<double(std::vector<double>)> &f,
    const std::function<std::vector<double>(std::vector<double>)> &grad,
    const std::vector<double> &gradient)
{

    const double C1 = 1e-4;
    const double C2 = 0.9;
    const double INITIAL_STEP_SIZE = 1.0;
    const double BACKTRACKING_ALPHA = 0.5;
    const double BACKTRACKING_TOL = 1e-10;

    double alpha = INITIAL_STEP_SIZE;
    int iter = 0;

    double f_current = f(x);
    double gradient_dot_d = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        gradient_dot_d += gradient[i] * d[i];
    }

    // Cache previously computed points to avoid redundant calculations
    std::unordered_map<double, std::pair<double, std::vector<double>>> cache;

    double alpha_lo = 0.0;
    double alpha_hi = std::numeric_limits<double>::max();

    while (iter++ < 20)
    {
        if (cache.find(alpha) != cache.end())
        {
            auto cached_result = cache[alpha];
            double f_new = cached_result.first;

            if (f_new <= f_current + C1 * alpha * gradient_dot_d)
            {
                std::vector<double> gradient_new = grad(cached_result.second);
                double gradient_new_dot_d = 0.0;
                for (size_t i = 0; i < x.size(); i++)
                {
                    gradient_new_dot_d += gradient_new[i] * d[i];
                }

                if (gradient_new_dot_d >= C2 * gradient_dot_d)
                {
                    break;
                }
                else
                {
                    alpha_lo = alpha;
                }
            }
            else
            {
                alpha_hi = alpha;
            }
        }
        else
        {
            std::vector<double> x_new(x.size());
            for (size_t i = 0; i < x.size(); i++)
            {
                x_new[i] = x[i] + alpha * d[i];
            }

            double f_new = f(x_new);
            cache[alpha] = std::make_pair(f_new, x_new);

            if (f_new <= f_current + C1 * alpha * gradient_dot_d)
            {
                std::vector<double> gradient_new = grad(x_new);
                double gradient_new_dot_d = 0.0;
                for (size_t i = 0; i < x.size(); i++)
                {
                    gradient_new_dot_d += gradient_new[i] * d[i];
                }

                if (gradient_new_dot_d >= C2 * gradient_dot_d)
                {
                    break;
                }
                else
                {
                    alpha_lo = alpha;
                }
            }
            else
            {
                alpha_hi = alpha;
            }
        }

        if (alpha_hi < std::numeric_limits<double>::max())
        {
            alpha = (alpha_lo + alpha_hi) / 2.0;
        }
        else
        {
            alpha = 2.0 * alpha_lo;
        }

        if (alpha < BACKTRACKING_TOL)
            break;
    }

    return alpha;
}

double armijoInterpolationLineSearch(
    const vector<double> &x,
    const vector<double> &d,
    const function<double(vector<double>)> &f,
    const vector<double> &gradient)
{
    const double f_x = f(x);
    const double grad_dot_d = dotProduct(gradient, d);
    double alpha = INITIAL_STEP_SIZE;
    const int N = x.size();
    vector<double> x_new(N);
    double alpha_prev = 0.0;
    double f_prev = f_x;

    int iteration = 0;
    const int max_iterations = 20; // Prevent infinite loops

    while (iteration++ < max_iterations)
    {
        for (int i = 0; i < N; ++i)
        {
            x_new[i] = x[i] + alpha * d[i];
        }

        double f_new = f(x_new);

        if (f_new <= f_x + C1 * alpha * grad_dot_d)
        {
            return alpha;
        }

        if (alpha < WOLFE_INTERP_MIN)
        {
            return WOLFE_INTERP_MIN;
        }

        if (alpha_prev > 0)
        {
            double delta_alpha = alpha - alpha_prev;
            if (abs(delta_alpha) < 1e-10)
            {
                alpha *= 0.5;
            }
            else
            {
                double grad_alpha = (f_new - f_x - grad_dot_d * alpha) / (alpha * alpha);
                alpha = cubicInterpolate(alpha_prev, alpha, f_prev, grad_dot_d,
                                         f_new, grad_alpha);

                if (alpha < 0.1 * alpha_prev || alpha > 0.9 * alpha_prev)
                {
                    alpha = alpha_prev * 0.5;
                }
            }
        }
        else
        {
            alpha = quadraticInterpolate(alpha, 0.0, f_new, grad_dot_d, f_x);
            if (alpha < 0.1 * INITIAL_STEP_SIZE || alpha > 0.9 * INITIAL_STEP_SIZE)
            {
                alpha = INITIAL_STEP_SIZE * 0.5;
            }
        }

        alpha_prev = alpha;
        f_prev = f_new;
    }
    if (alpha < 1e-4)
    {
        return 0.5; // Prevent excessively small steps
    }
    return alpha;
}

// Safe cubic interpolation with bounds and numerical checks
double safeCubicInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1, double dphi1)
{
    if (alpha0 > alpha1)
    {
        std::swap(alpha0, alpha1);
        std::swap(phi0, phi1);
        std::swap(dphi0, dphi1);
    }

    double d1;
    try
    {
        d1 = dphi0 + dphi1 - 3 * (phi1 - phi0) / (alpha1 - alpha0);
    }
    catch (...)
    {
        return 0.5 * (alpha0 + alpha1);
    }

    if (std::isnan(d1) || std::isinf(d1))
    {
        return 0.5 * (alpha0 + alpha1);
    }

    double discriminant = d1 * d1 - dphi0 * dphi1;

    if (discriminant < 0)
    {
        return 0.5 * (alpha0 + alpha1);
    }

    double d2;
    try
    {
        d2 = std::copysign(sqrt(discriminant), alpha1 - alpha0);
    }
    catch (...)
    {
        return 0.5 * (alpha0 + alpha1);
    }

    double denominator = dphi0 - dphi1 + 2 * d2;

    if (std::abs(denominator) < 1e-10)
    {
        return 0.5 * (alpha0 + alpha1);
    }

    double result;
    try
    {
        result = alpha0 + (alpha1 - alpha0) * (dphi0 + d2 - d1) / denominator;
    }
    catch (...)
    {
        return 0.5 * (alpha0 + alpha1);
    }

    if (std::isnan(result) || std::isinf(result))
    {
        return 0.5 * (alpha0 + alpha1);
    }

    return std::max(alpha0 + 0.1 * (alpha1 - alpha0),
                    std::min(alpha1 - 0.1 * (alpha1 - alpha0), result));
}

double wolfeInterpolationLineSearch(
    const vector<double> &x,
    const vector<double> &d,
    const function<double(vector<double>)> &f,
    const function<vector<double>(vector<double>)> &grad,
    const vector<double> &gradient)
{
    const double f_x = f(x);
    const double grad_dot_d = dotProduct(gradient, d);
    double alpha = INITIAL_STEP_SIZE;
    const int N = x.size();
    vector<double> x_new(N);

    double alpha_lo = 0.0;
    double alpha_hi = numeric_limits<double>::infinity();
    double f_lo = f_x;
    double dphi_lo = grad_dot_d;

    for (int iter = 0; iter < 20; ++iter)
    {
        for (int i = 0; i < N; ++i)
        {
            x_new[i] = x[i] + alpha * d[i];
        }

        double f_new = f(x_new);

        if (f_new > f_x + C1 * alpha * grad_dot_d || (f_new >= f_lo && iter > 0))
        {
            alpha_hi = alpha;
            alpha = safeCubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, f_new,
                                         (f_new - f_x - grad_dot_d * alpha) / (alpha * alpha));
            continue;
        }

        vector<double> grad_new = grad(x_new);
        double dphi_new = dotProduct(grad_new, d);

        if (abs(dphi_new) <= -C2 * grad_dot_d)
        {
            return alpha;
        }

        if (dphi_new >= 0)
        {
            alpha_hi = alpha;
            alpha = safeCubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, f_new, dphi_new);
        }
        else
        {
            alpha_lo = alpha;
            f_lo = f_new;
            dphi_lo = dphi_new;

            if (alpha_hi == numeric_limits<double>::infinity())
            {
                alpha *= 2;
            }
            else
            {
                alpha = safeCubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, f_new, dphi_new);
            }
        }

        if (alpha < WOLFE_INTERP_MIN)
        {
            return WOLFE_INTERP_MIN;
        }
    }

    return alpha;
}