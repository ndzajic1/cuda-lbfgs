#include "LineSearch.h"
#include "LBFGSConstants.h"
#include "VectorOps.h"
#include <limits>
using namespace std;

// Helper functions for interpolation
double cubicInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1, double dphi1) {
    double d1 = dphi0 + dphi1 - 3*(phi1 - phi0)/(alpha1 - alpha0);
    double d2 = std::copysign(sqrt(d1*d1 - dphi0*dphi1), alpha1 - alpha0);
    return alpha0 + (alpha1 - alpha0)*(dphi0 + d2 - d1)/(dphi0 - dphi1 + 2*d2);
}

double quadraticInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1) {
    return alpha0 - 0.5*dphi0*alpha0*alpha0/(phi1 - phi0 - dphi0*alpha0);
}

// Backtracking line search (Armijo condition)
double backtrackingLineSearch(const vector<double>& x, const vector<double>& d,
                              const function<double(vector<double>)>& f, 
                              const vector<double>& gradient) {
    double alpha = INITIAL_STEP_SIZE;
    
    while (f(x) - f(add(x, scalarProduct(alpha, d))) < C1 * alpha * dotProduct(gradient, d)) {
        alpha *= BACKTRACKING_ALPHA;
        if (alpha < BACKTRACKING_TOL) break;
    }
    
    return alpha;
}

// Backtracking Wolfe line search (Armijo + Wolfe conditions)
double backtrackingWolfeLineSearch(const vector<double>& x, const vector<double>& d,
                                   const function<double(vector<double>)>& f, 
                                   const function<vector<double>(vector<double>)>& grad, 
                                   const vector<double>& gradient) {
    double alpha = INITIAL_STEP_SIZE;

    while (true) {
        vector<double> x_new = add(x, scalarProduct(alpha, d));
        vector<double> gradient_new = grad(x_new);

        if (f(x_new) > f(x) + C1 * alpha * dotProduct(gradient, d)) {
            alpha *= BACKTRACKING_ALPHA;  // Reduce step size
        } else if (dotProduct(gradient_new, d) < C2 * dotProduct(gradient, d)) {
            alpha *= 1.1;  // Increase step size if Wolfe condition isn't met
        } else {
            break;
        }

        if (alpha < BACKTRACKING_TOL) break;
    }

    return alpha;
}

double armijoInterpolationLineSearch(
    const vector<double>& x,
    const vector<double>& d,
    const function<double(vector<double>)>& f,
    const vector<double>& gradient
) {
    const double f_x = f(x);
    const double grad_dot_d = dotProduct(gradient, d);
    double alpha = INITIAL_STEP_SIZE;
    const int N = x.size();
    vector<double> x_new(N);
    double alpha_prev = 0.0;
    double f_prev = f_x;
    
    int iteration = 0;
    const int max_iterations = 20;  // Prevent infinite loops
    
    while (iteration++ < max_iterations) {
        // Compute new point
        for (int i = 0; i < N; ++i) {
            x_new[i] = x[i] + alpha * d[i];
        }
        
        double f_new = f(x_new);
        
        // Check Armijo condition
        if (f_new <= f_x + C1 * alpha * grad_dot_d) {
            return alpha;
        }
        
        if (alpha < WOLFE_INTERP_MIN) {
            return WOLFE_INTERP_MIN;
        }
        
        // Cubic interpolation if we have two points
        if (alpha_prev > 0) {
            // Add safeguard for numerical stability
            double delta_alpha = alpha - alpha_prev;
            if (abs(delta_alpha) < 1e-10) {
                alpha *= 0.5;  // fallback to simple backtracking
            } else {
                double grad_alpha = (f_new - f_x - grad_dot_d * alpha)/(alpha * alpha);
                alpha = cubicInterpolate(alpha_prev, alpha, f_prev, grad_dot_d, 
                                       f_new, grad_alpha);
                
                // Safeguard the interpolated value
                if (alpha < 0.1 * alpha_prev || alpha > 0.9 * alpha_prev) {
                    alpha = alpha_prev * 0.5;  // fallback to simple backtracking
                }
            }
        } else {
            // Quadratic interpolation for first step
            alpha = quadraticInterpolate(alpha, 0.0, f_new, grad_dot_d, f_x);
            // Safeguard the interpolated value
            if (alpha < 0.1 * INITIAL_STEP_SIZE || alpha > 0.9 * INITIAL_STEP_SIZE) {
                alpha = INITIAL_STEP_SIZE * 0.5;
            }
        }
        
        alpha_prev = alpha;
        f_prev = f_new;
    }
    
    return alpha;  // Return best found if max iterations reached
}


// Wolfe conditions with cubic interpolation
double wolfeInterpolationLineSearch(
    const vector<double>& x, 
    const vector<double>& d,
    const function<double(vector<double>)>& f, 
    const function<vector<double>(vector<double>)>& grad, 
    const vector<double>& gradient
) {
    const double f_x = f(x);
    const double grad_dot_d = dotProduct(gradient, d);
    double alpha = INITIAL_STEP_SIZE;  // Using constant from LBFGSConstants.h
    const int N = x.size();
    vector<double> x_new(N);
    
    double alpha_lo = 0.0;
    double alpha_hi = numeric_limits<double>::infinity();
    double f_lo = f_x;
    double dphi_lo = grad_dot_d;
    
    for (int iter = 0; iter < 20; ++iter) {  // max iterations
        // Evaluate current point
        for (int i = 0; i < N; ++i) {
            x_new[i] = x[i] + alpha * d[i];
        }
        
        double f_new = f(x_new);
        
        // Check Armijo condition
        if (f_new > f_x + C1 * alpha * grad_dot_d || (f_new >= f_lo && iter > 0)) {  // Using C1 constant for Armijo condition
            alpha_hi = alpha;
            alpha = cubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, f_new, 
                                   (f_new - f_x - grad_dot_d * alpha)/(alpha * alpha));
            continue;
        }
        
        // Evaluate gradient at new point
        vector<double> grad_new = grad(x_new);
        double dphi_new = dotProduct(grad_new, d);
        
        // Check Wolfe condition
        if (abs(dphi_new) <= -C2 * grad_dot_d) {  // Using C2 constant for Wolfe condition
            return alpha;
        }
        
        if (dphi_new >= 0) {
            alpha_hi = alpha;
            alpha = cubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, f_new, dphi_new);
        } else {
            alpha_lo = alpha;
            f_lo = f_new;
            dphi_lo = dphi_new;
            
            if (alpha_hi == numeric_limits<double>::infinity()) {
                alpha *= 2;
            } else {
                alpha = cubicInterpolate(alpha_lo, alpha_hi, f_lo, dphi_lo, f_new, dphi_new);
            }
        }
        
        if (alpha < WOLFE_INTERP_MIN) {  // Using WOLFE_INTERP_MIN constant
            return WOLFE_INTERP_MIN;
        }
    }
    
    return alpha;  // Return best found if max iterations reached
}