/*
Code converted from Python to C using GPT-4o (by OpenAI)

> Original Code is written in Python!

> If you find bugs or have suggestions, please open an issue on GitHub.

Date: 10.05.2025
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define N_SAMPLES 3
#define N_FEATURES 4
#define N_BASIS_FUNCTIONS 6
#define N_CLASSES 3
#define MAX_BASIS_LEN 20

typedef struct {
    char basis_functions[N_BASIS_FUNCTIONS][MAX_BASIS_LEN];
    double coefficients_list[N_CLASSES][N_BASIS_FUNCTIONS];
    int n_features;
    int n_classes;
} SymbolicModel;

// Softmax over logits
void softmax(double logits[N_SAMPLES][N_CLASSES], double probs[N_SAMPLES][N_CLASSES]) {
    for (int i = 0; i < N_SAMPLES; i++) {
        double max_val = logits[i][0];
        for (int j = 1; j < N_CLASSES; j++) {
            if (logits[i][j] > max_val) {
                max_val = logits[i][j];
            }
        }

        double sum_exp = 0.0;
        for (int j = 0; j < N_CLASSES; j++) {
            probs[i][j] = exp(logits[i][j] - max_val);
            sum_exp += probs[i][j];
        }

        for (int j = 0; j < N_CLASSES; j++) {
            probs[i][j] /= sum_exp;
        }
    }
}

// Evaluate symbolic basis functions
void evaluate_basis_functions(double X[N_SAMPLES][N_FEATURES], SymbolicModel *model,
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
                int idx, power;
                sscanf(func, "x%d^%d", &idx, &power);
                X_transformed[j][i] = pow(X[j][idx], power);
            } else if (strchr(func, ' ') != NULL) {
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

// Predict class labels
void predict(double X[N_SAMPLES][N_FEATURES], SymbolicModel *model, int y_pred[N_SAMPLES]) {
    double X_transformed[N_SAMPLES][N_BASIS_FUNCTIONS];
    double logits[N_SAMPLES][N_CLASSES];
    double probs[N_SAMPLES][N_CLASSES];

    evaluate_basis_functions(X, model, X_transformed);

    // Compute logits
    for (int i = 0; i < N_SAMPLES; i++) {
        for (int k = 0; k < N_CLASSES; k++) {
            logits[i][k] = 0.0;
            for (int j = 0; j < N_BASIS_FUNCTIONS; j++) {
                logits[i][k] += X_transformed[i][j] * model->coefficients_list[k][j];
            }
        }
    }

    softmax(logits, probs);

    // Argmax for prediction
    for (int i = 0; i < N_SAMPLES; i++) {
        double max_prob = probs[i][0];
        int max_idx = 0;
        for (int k = 1; k < N_CLASSES; k++) {
            if (probs[i][k] > max_prob) {
                max_prob = probs[i][k];
                max_idx = k;
            }
        }
        y_pred[i] = max_idx;
    }
}

int main() {
    // Define symbolic model
    SymbolicModel model = {
        .basis_functions = { "1", "x0", "x1^2", "x2", "log1p_x3", "x1 x2" },
        .coefficients_list = {
            {0.5, 1.2, -0.8, 0.1, 0.3, -0.2},
            {-0.3, 0.6, 0.5, -0.4, 0.7, 0.1},
            {0.1, -1.0, 0.3, 0.9, -0.6, 0.5}
        },
        .n_features = N_FEATURES,
        .n_classes = N_CLASSES
    };

    double X[N_SAMPLES][N_FEATURES] = {
        {5.1, 3.5, 1.4, 0.2},
        {7.0, 3.2, 4.7, 1.4},
        {6.3, 3.3, 6.0, 2.5}
    };

    int y_pred[N_SAMPLES];
    predict(X, &model, y_pred);

    for (int i = 0; i < N_SAMPLES; i++) {
        printf("Predicted class for sample %d: %d\n", i, y_pred[i]);
    }

    return 0;
}