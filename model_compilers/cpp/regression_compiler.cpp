/*
Code converted from Python to C++ using GPT-4o (by OpenAI)

> Original Code is written in Python!

> If you find bugs or have suggestions, please open an issue on GitHub.

Date: 10.05.2025
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;
using namespace Eigen;

vector<string> basis_functions;
vector<double> coefficients;
int n_features;

double apply_basis_function(const VectorXd& x, const string& func) {
    if (func == "1") return 1.0;
    else if (func.rfind("log1p_x", 0) == 0) {
        int idx = stoi(func.substr(8));
        return log1p(abs(x[idx]));
    } else if (func.rfind("exp_x", 0) == 0) {
        int idx = stoi(func.substr(5));
        return exp(min(max(x[idx], -10.0), 10.0));
    } else if (func.rfind("sin_x", 0) == 0) {
        int idx = stoi(func.substr(5));
        return sin(x[idx]);
    } else if (func.find('^') != string::npos) {
        size_t pos = func.find('^');
        int idx = stoi(func.substr(1, pos - 1));
        int power = stoi(func.substr(pos + 1));
        return pow(x[idx], power);
    } else if (func.find(' ') != string::npos) {
        double result = 1.0;
        size_t start = 0, end;
        while ((end = func.find(' ', start)) != string::npos) {
            string token = func.substr(start, end - start);
            result *= x[stoi(token.substr(1))];
            start = end + 1;
        }
        result *= x[stoi(func.substr(start).substr(1))];
        return result;
    } else {
        int idx = stoi(func.substr(1));
        return x[idx];
    }
}

VectorXd predict(const MatrixXd& X) {
    int n_samples = X.rows();
    int n_basis = basis_functions.size();
    MatrixXd X_transformed(n_samples, n_basis);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_basis; ++j) {
            X_transformed(i, j) = apply_basis_function(X.row(i), basis_functions[j]);
        }
    }

    VectorXd coeffs = Map<VectorXd>(coefficients.data(), coefficients.size());
    return X_transformed * coeffs;
}

int main() {
    // Load model from JSON
    ifstream in("outputs/california_housing_model.json");
    json symbolic_model;
    in >> symbolic_model;
    basis_functions = symbolic_model["basis_functions"].get<vector<string>>();
    coefficients = symbolic_model["coefficients"].get<vector<double>>();
    n_features = symbolic_model["n_features"].get<int>();

    // Generate input data
    MatrixXd X = MatrixXd::Random(10, n_features);
    VectorXd y_pred = predict(X);
    cout << "Predictions:\n" << y_pred << endl;
    return 0;
}
