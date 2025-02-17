#ifndef VECTOR__H
#define VECTOR_OPS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace std;

void printMatrix(const vector<vector<double>> &matrix);
void printVector(const vector<double> &vector);
void ensureSameSize(const vector<double> &v1, const vector<double> &v2);
double dotProduct(const vector<double> &v1, const vector<double> &v2);
vector<double> scalarProduct(const double scalar, const vector<double> &v);
vector<double> add(const vector<double> &v1, const vector<double> &v2);
vector<double> negative(const vector<double> &v);
double vectorNorm(const vector<double> &v);
double getRho(const vector<double> &s, const vector<double> &y);

#endif // VECTOR_OPS_H
