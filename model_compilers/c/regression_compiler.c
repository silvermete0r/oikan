/*
Code converted from Python to C using GPT-4o (by OpenAI)

> Original Code is written in Python!

> If you find bugs or have suggestions, please open an issue on GitHub.

Date: 10.05.2025
*/

#include <stdio.h>
#include <string.h>
#include <math.h>

#define N_SAMPLES 10
#define N_FEATURES 8
#define N_BASIS_FUNCTIONS 5
#define MAX_BASIS_LEN 20

// Example symbolic model structure
typedef struct {
    char basis_functions[N_BASIS_FUNCTIONS][MAX_BASIS_LEN];
    double coefficients[N_BASIS_FUNCTIONS];
    int n_features;
} SymbolicModel;

// Evaluates the basis functions
void evaluate_basis_functions(double X[N_SAMPLES][N_FEATURES],
                              SymbolicModel *model,
                              double X_transformed[N_SAMPLES][N_BASIS_FUNCTIONS]) {
    for (int i = 0; i < N_BASIS_FUNCTIONS; i++) {
        char *func = model->basis_functions[i];

        for (int j = 0; j < N_SAMPLES; j++) {
            if (strcmp(func, "1") == 0) {
                X_transformed[j][i] = 1.0;
            } else if (strncmp(func, "log1p_x", 7) == 0) {
                int idx = atoi(&func[7]);
                X_transformed[j][i] = log1p(fabs(X[j][idx]));
            } else if (strncmp(func, "exp_x", 5) == 0) {
                int idx = atoi(&func[5]);
                double clipped = fmax(fmin(X[j][idx], 10.0), -10.0);
                X_transformed[j][i] = exp(clipped);
            } else if (strncmp(func, "sin_x", 5) == 0) {
                int idx = atoi(&func[5]);
                X_transformed[j][i] = sin(X[j][idx]);
            } else if (strchr(func, '^') != NULL) {
                char var[10];
                int idx, power;
                sscanf(func, "x%d^%d", &idx, &power);
                double val = X[j][idx];
                X_transformed[j][i] = pow(val, power);
            } else if (strchr(func, ' ') != NULL) {
                // Multiple variables multiplied, like "x0 x1"
                char *token;
                char func_copy[MAX_BASIS_LEN];
                strcpy(func_copy, func);
                token = strtok(func_copy, " ");
                double result = 1.0;
                while (token != NULL) {
                    int idx = atoi(&token[1]);
                    result *= X[j][idx];
                    token = strtok(NULL, " ");
                }
                X_transformed[j][i] = result;
            } else {
                int idx = atoi(&func[1]);
                X_transformed[j][i] = X[j][idx];
            }
        }
    }
}

// Performs prediction
void predict(double X[N_SAMPLES][N_FEATURES], SymbolicModel *model, double y_pred[N_SAMPLES]) {
    double X_transformed[N_SAMPLES][N_BASIS_FUNCTIONS];
    evaluate_basis_functions(X, model, X_transformed);

    for (int i = 0; i < N_SAMPLES; i++) {
        y_pred[i] = 0.0;
        for (int j = 0; j < N_BASIS_FUNCTIONS; j++) {
            y_pred[i] += X_transformed[i][j] * model->coefficients[j];
        }
    }
}

int main() {
    // Example symbolic model
    SymbolicModel model = {
        .basis_functions = { "1", "x0", "x1^2", "x2 x3", "log1p_x4" },
        .coefficients = { 1.0, 0.5, -0.2, 0.1, 0.3 },
        .n_features = N_FEATURES
    };

    // Random input data
    double X[N_SAMPLES][N_FEATURES];
    for (int i = 0; i < N_SAMPLES; i++) {
        for (int j = 0; j < N_FEATURES; j++) {
            X[i][j] = (double)rand() / RAND_MAX;
        }
    }

    double y_pred[N_SAMPLES];
    predict(X, &model, y_pred);

    for (int i = 0; i < N_SAMPLES; i++) {
        printf("Prediction %d: %f\n", i, y_pred[i]);
    }

    return 0;
}