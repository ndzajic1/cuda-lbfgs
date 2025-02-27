#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H

#include <iostream>
#include <vector>

using namespace std;

void ensureSameSize(const vector<double> &v1, const vector<double> &v2);

double dotProduct(const vector<double> &v1, const vector<double> &v2);

vector<double> scalarProduct(const double scalar, const vector<double> &v);

vector<double> add(const vector<double> &v1, const vector<double> &v2);

#endif