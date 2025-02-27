#include "LBFGSConfig.h"
#include "LineSearch.h"

LineSearchFunction selectLineSearch(LineSearchMethod method) {
    switch (method) {
        case LineSearchMethod::Backtracking:
            return backtrackingLineSearch;
        case LineSearchMethod::ArmijoInterpolation:
            return armijoInterpolationLineSearch;
        default:
            throw invalid_argument("Invalid line search method for function without gradient.");
    }
}

LineSearchFunctionWithGradient selectLineSearchWithGradient(LineSearchMethod method) {
    switch (method) {
        case LineSearchMethod::BacktrackingWolfe:
            return backtrackingWolfeLineSearch;
        case LineSearchMethod::WolfeInterpolation:
            return wolfeInterpolationLineSearch;
        default:
            throw invalid_argument("Invalid line search method for function with gradient.");
    }
}
