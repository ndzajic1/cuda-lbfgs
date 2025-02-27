#ifndef CONSTANTS_H
#define CONSTANTS_H

// Armijo and Wolfe condition constants
constexpr double C1 = 1e-4;   // Armijo condition constant
constexpr double C2 = 0.7;    // Wolfe condition constant

// Initial step size for line search
constexpr double INITIAL_STEP_SIZE = 1.0;

// Backtracking parameters
constexpr double BACKTRACKING_ALPHA = 0.5;  // Step reduction factor
constexpr double BACKTRACKING_TOL = 1e-8;  // Convergence threshold

// Wolfe interpolation parameters
constexpr double WOLFE_INTERP_MIN = 1e-10;
constexpr double WOLFE_INTERP_MAX = 10.0;


constexpr double MAX_STEP_SIZE = 10.0;
constexpr double MIN_STEP_SIZE = 1e-6;

#endif // CONSTANTS_H