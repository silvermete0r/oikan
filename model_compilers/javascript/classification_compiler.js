/*
Code converted from Python to JavaScript using GPT-4o (by OpenAI)

> Original Code is written in Python!

> If you find bugs or have suggestions, please open an issue on GitHub.

Date: 10.05.2025
*/

function softmax(logits) {
    const maxes = logits.map(row => Math.max(...row));
    const exps = logits.map((row, i) =>
        row.map(val => Math.exp(val - maxes[i]))
    );
    const sums = exps.map(row => row.reduce((a, b) => a + b, 0));
    return exps.map((row, i) => row.map(val => val / sums[i]));
}

function evaluateBasisFunctions(X, basisFunctions, nFeatures) {
    const nSamples = X.length;
    const XTransformed = Array.from({ length: nSamples }, () =>
        Array(basisFunctions.length).fill(0)
    );

    for (let i = 0; i < basisFunctions.length; i++) {
        const func = basisFunctions[i];

        for (let j = 0; j < nSamples; j++) {
            if (func === '1') {
                XTransformed[j][i] = 1;
            } else if (func.startsWith('log1p_x')) {
                const idx = parseInt(func.split('_')[1].substring(1));
                XTransformed[j][i] = Math.log1p(Math.abs(X[j][idx]));
            } else if (func.startsWith('exp_x')) {
                const idx = parseInt(func.split('_')[1].substring(1));
                const val = Math.max(-10, Math.min(10, X[j][idx]));
                XTransformed[j][i] = Math.exp(val);
            } else if (func.startsWith('sin_x')) {
                const idx = parseInt(func.split('_')[1].substring(1));
                XTransformed[j][i] = Math.sin(X[j][idx]);
            } else if (func.includes('^')) {
                const [varName, power] = func.split('^');
                const idx = parseInt(varName.substring(1));
                XTransformed[j][i] = Math.pow(X[j][idx], parseInt(power));
            } else if (func.includes(' ')) {
                const vars = func.split(' ');
                let result = 1;
                for (const v of vars) {
                    const idx = parseInt(v.substring(1));
                    result *= X[j][idx];
                }
                XTransformed[j][i] = result;
            } else {
                const idx = parseInt(func.substring(1));
                XTransformed[j][i] = X[j][idx];
            }
        }
    }

    return XTransformed;
}

function dotMatrix(A, B) {
    const result = Array.from({ length: A.length }, () =>
        Array(B[0].length).fill(0)
    );
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B[0].length; j++) {
            for (let k = 0; k < B.length; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

function predict(X, symbolicModel) {
    const XTransformed = evaluateBasisFunctions(X, symbolicModel.basis_functions, symbolicModel.n_features);
    const coeffs = symbolicModel.coefficients_list[0] instanceof Array
        ? symbolicModel.coefficients_list
        : [symbolicModel.coefficients_list];

    // Transpose coefficients
    const coeffsT = coeffs[0].map((_, colIndex) => coeffs.map(row => row[colIndex]));
    const logits = dotMatrix(XTransformed, coeffsT);
    const probabilities = softmax(logits);

    return probabilities.map(row => row.indexOf(Math.max(...row)));
}

// Example usage
fetch('outputs/iris_model.json')
    .then(response => response.json())
    .then(symbolicModel => {
        const X = [
            [5.1, 3.5, 1.4, 0.2],
            [7.0, 3.2, 4.7, 1.4],
            [6.3, 3.3, 6.0, 2.5]
        ];
        const yPred = predict(X, symbolicModel);
        console.log(yPred);
    });
