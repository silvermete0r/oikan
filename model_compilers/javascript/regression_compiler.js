/*
Code converted from Python to JavaScript using GPT-4o (by OpenAI)

> Original Code is written in Python!

> If you find bugs or have suggestions, please open an issue on GitHub.

Date: 10.05.2025
*/

function evaluateBasisFunctions(X, basisFunctions, nFeatures) {
    const nSamples = X.length;
    const XTransformed = Array.from({ length: nSamples }, () => Array(basisFunctions.length).fill(0));

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

function predict(X, symbolicModel) {
    const XTransformed = evaluateBasisFunctions(X, symbolicModel.basis_functions, symbolicModel.n_features);
    const yPred = XTransformed.map(row =>
        row.reduce((sum, val, idx) => sum + val * symbolicModel.coefficients[idx], 0)
    );
    return yPred;
}

// Example usage:
fetch('outputs/california_housing_model.json')
    .then(response => response.json())
    .then(symbolicModel => {
        const nSamples = 10;
        const nFeatures = symbolicModel.n_features;
        const X = Array.from({ length: nSamples }, () =>
            Array.from({ length: nFeatures }, () => Math.random())
        );
        const yPred = predict(X, symbolicModel);
        console.log(yPred);
    });
