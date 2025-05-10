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
vector<vector<double>> coefficients_list;
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

MatrixXd softmax(const MatrixXd& logits) {
    MatrixXd stabilized = logits.rowwise() - logits.rowwise().maxCoeff();
    MatrixXd exp_logits = stabilized.array().exp();
    VectorXd row_sums = exp_logits.rowwise().sum();
    return exp_logits.array().colwise() / row_sums.array();
}

VectorXi predict(const MatrixXd& X) {
    int n_samples = X.rows();
    int n_basis = basis_functions.size();
    int n_classes = coefficients_list.size();

    MatrixXd X_transformed(n_samples, n_basis);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_basis; ++j) {
            X_transformed(i, j) = apply_basis_function(X.row(i), basis_functions[j]);
        }
    }

    MatrixXd coef_matrix(n_basis, n_classes);
    for (int c = 0; c < n_classes; ++c) {
        coef_matrix.col(c) = Map<VectorXd>(coefficients_list[c].data(), n_basis);
    }

    MatrixXd logits = X_transformed * coef_matrix;
    MatrixXd probs = softmax(logits);

    VectorXi predictions(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        RowVectorXd row = probs.row(i);
        row.maxCoeff(&predictions[i]);
    }
    return predictions;
}

int main() {
    ifstream in("outputs/iris_model.json");
    json symbolic_model;
    in >> symbolic_model;

    basis_functions = symbolic_model["basis_functions"].get<vector<string>>();
    coefficients_list = symbolic_model["coefficients_list"].get<vector<vector<double>>>();
    n_features = symbolic_model["n_features"].get<int>();

    MatrixXd X(3, n_features);
    X << 5.1, 3.5, 1.4, 0.2,
         7.0, 3.2, 4.7, 1.4,
         6.3, 3.3, 6.0, 2.5;

    VectorXi y_pred = predict(X);
    cout << "Predicted class indices:\n" << y_pred.transpose() << endl;
    return 0;
}