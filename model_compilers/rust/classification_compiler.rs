/*
Code converted from Python to Rust using GPT-4o (by OpenAI)

> Original Code is written in Python!

> If you find bugs or have suggestions, please open an issue on GitHub.

Date: 10.05.2025
*/

use ndarray::{Array2, Array1, Axis};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
struct SymbolicModel {
    basis_functions: Vec<String>,
    coefficients_list: Vec<Vec<f64>>,
    n_features: usize,
}

fn softmax(X: &Array2<f64>) -> Array2<f64> {
    let max_vals = X.fold_axis(Axis(1), f64::NEG_INFINITY, |a, &b| a.max(b));
    let e_X = X - &max_vals.insert_axis(Axis(1)); // Avoid overflow with max trick
    let e_X = e_X.mapv(f64::exp);
    let sum_e_X = e_X.sum_axis(Axis(1)).insert_axis(Axis(1));
    e_X / sum_e_X
}

fn evaluate_basis_functions(X: &Array2<f64>, basis_functions: &[String], n_features: usize) -> Array2<f64> {
    let n_samples = X.nrows();
    let mut X_transformed = Array2::<f64>::zeros((n_samples, basis_functions.len()));

    for (i, func) in basis_functions.iter().enumerate() {
        let col = X_transformed.column_mut(i);
        let values: Array1<f64> = match func.as_str() {
            "1" => Array1::ones(n_samples),
            f if f.starts_with("log1p_x") => {
                let idx: usize = f[7..].parse().unwrap();
                X.column(idx).mapv(|v| (v.abs() + 1.0).ln())
            }
            f if f.starts_with("exp_x") => {
                let idx: usize = f[5..].parse().unwrap();
                X.column(idx).mapv(|v| (v.max(-10.0).min(10.0)).exp())
            }
            f if f.starts_with("sin_x") => {
                let idx: usize = f[5..].parse().unwrap();
                X.column(idx).mapv(f64::sin)
            }
            f if f.contains('^') => {
                let parts: Vec<&str> = f.split('^').collect();
                let idx: usize = parts[0][1..].parse().unwrap();
                let power: u32 = parts[1].parse().unwrap();
                X.column(idx).mapv(|v| v.powi(power as i32))
            }
            f if f.contains(' ') => {
                let vars: Vec<&str> = f.split_whitespace().collect();
                let mut result = Array1::<f64>::ones(n_samples);
                for var in vars {
                    let idx: usize = var[1..].parse().unwrap();
                    result = result * &X.column(idx);
                }
                result
            }
            f => {
                let idx: usize = f[1..].parse().unwrap();
                X.column(idx).to_owned()
            }
        };
        col.assign(&values);
    }

    X_transformed
}

fn predict(X: &Array2<f64>, model: &SymbolicModel) -> Array1<usize> {
    let X_transformed = evaluate_basis_functions(X, &model.basis_functions, model.n_features);
    let logits = X_transformed.dot(&Array2::from(model.coefficients_list.clone()).reversed_axes());
    let probabilities = softmax(&logits);
    probabilities.mapv(|x| x.argmax().unwrap())
}

fn main() {
    let file = File::open("outputs/iris_model.json").unwrap();
    let reader = BufReader::new(file);
    let symbolic_model: SymbolicModel = serde_json::from_reader(reader).unwrap();

    let X = Array2::<f64>::from_shape_vec((3, 4), vec![5.1, 3.5, 1.4, 0.2,
                                                     7.0, 3.2, 4.7, 1.4,
                                                     6.3, 3.3, 6.0, 2.5]).unwrap();

    let y_pred = predict(&X, &symbolic_model);
    println!("{:?}", y_pred);
}