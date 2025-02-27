#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include <vector>
#include <functional>

using namespace std;

// Function prototypes for line search methods
double backtrackingLineSearch(const vector<double>& x, const vector<double>& d,
                              const function<double(vector<double>)>& f, 
                              const vector<double>& gradient);

double backtrackingWolfeLineSearch(const vector<double>& x, const vector<double>& d,
                                   const function<double(vector<double>)>& f, 
                                   const function<vector<double>(vector<double>)>& grad, 
                                   const vector<double>& gradient);

double armijoInterpolationLineSearch(const vector<double>& x, const vector<double>& d,
                                     const function<double(vector<double>)>& f, 
                                     const vector<double>& gradient);

double wolfeInterpolationLineSearch(const vector<double>& x, const vector<double>& d,
                                    const function<double(vector<double>)>& f, 
                                    const function<vector<double>(vector<double>)>& grad, 
                                    const vector<double>& gradient);

#endif // LINE_SEARCH_H