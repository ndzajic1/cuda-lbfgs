#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include <iostream>
#include "constants.h"
#include <vector>
#include "vector_utils.h"
#include "functional"
#include "limits"
#include <cmath>

using namespace std;

// Helper functions for interpolation
double cubicInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1, double dphi1);

double quadraticInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1);

// Backtracking line search (Armijo condition)
double backtrackingLineSearch(const vector<double> &x, const vector<double> &d,
                              const function<double(vector<double>)> &f,
                              const vector<double> &gradient);

double backtrackingWolfeLineSearch(
    const std::vector<double> &x,
    const std::vector<double> &d,
    const std::function<double(std::vector<double>)> &f,
    const std::function<std::vector<double>(std::vector<double>)> &grad,
    const std::vector<double> &gradient);

double armijoInterpolationLineSearch(
    const vector<double> &x,
    const vector<double> &d,
    const function<double(vector<double>)> &f,
    const vector<double> &gradient);

// Safe cubic interpolation with bounds and numerical checks
double safeCubicInterpolate(double alpha0, double alpha1, double phi0, double dphi0, double phi1, double dphi1);

double wolfeInterpolationLineSearch(
    const vector<double> &x,
    const vector<double> &d,
    const function<double(vector<double>)> &f,
    const function<vector<double>(vector<double>)> &grad,
    const vector<double> &gradient);

#endif