/*
Code converted from Python to Go using GPT-4o (by OpenAI)

> Original Code is written in Python!

> If you find bugs or have suggestions, please open an issue on GitHub.

Date: 10.05.2025
*/

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

type SymbolicModel struct {
	BasisFunctions   []string    `json:"basis_functions"`
	NFeatures        int         `json:"n_features"`
	CoefficientsList [][]float64 `json:"coefficients_list"` // shape: n_classes x n_basis_functions
}

func softmax(X [][]float64) [][]float64 {
	nSamples := len(X)
	nClasses := len(X[0])
	result := make([][]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		maxVal := X[i][0]
		for _, v := range X[i] {
			if v > maxVal {
				maxVal = v
			}
		}
		expSum := 0.0
		result[i] = make([]float64, nClasses)
		for j := 0; j < nClasses; j++ {
			result[i][j] = math.Exp(X[i][j] - maxVal)
			expSum += result[i][j]
		}
		for j := 0; j < nClasses; j++ {
			result[i][j] /= expSum
		}
	}
	return result
}

func evaluateBasisFunctions(X [][]float64, basisFunctions []string, nFeatures int) [][]float64 {
	nSamples := len(X)
	nBasis := len(basisFunctions)
	XTransformed := make([][]float64, nSamples)
	for i := range XTransformed {
		XTransformed[i] = make([]float64, nBasis)
	}

	for j, funcStr := range basisFunctions {
		for i := 0; i < nSamples; i++ {
			val := 0.0
			switch {
			case funcStr == "1":
				val = 1.0
			case strings.HasPrefix(funcStr, "log1p_x"):
				idx := parseIndex(funcStr[len("log1p_x"):])
				val = math.Log1p(math.Abs(X[i][idx]))
			case strings.HasPrefix(funcStr, "exp_x"):
				idx := parseIndex(funcStr[len("exp_x"):])
				clipped := math.Min(math.Max(X[i][idx], -10), 10)
				val = math.Exp(clipped)
			case strings.HasPrefix(funcStr, "sin_x"):
				idx := parseIndex(funcStr[len("sin_x"):])
				val = math.Sin(X[i][idx])
			case strings.Contains(funcStr, "^"):
				parts := strings.Split(funcStr, "^")
				idx := parseIndex(parts[0][1:])
				power := parseIndex(parts[1])
				val = math.Pow(X[i][idx], float64(power))
			case strings.Contains(funcStr, " "):
				val = 1.0
				vars := strings.Split(funcStr, " ")
				for _, v := range vars {
					idx := parseIndex(v[1:])
					val *= X[i][idx]
				}
			default:
				idx := parseIndex(funcStr[1:])
				val = X[i][idx]
			}
			XTransformed[i][j] = val
		}
	}
	return XTransformed
}

func predict(X [][]float64, model SymbolicModel) []int {
	XT := evaluateBasisFunctions(X, model.BasisFunctions, model.NFeatures)
	nSamples := len(XT)
	nClasses := len(model.CoefficientsList)

	logits := make([][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		logits[i] = make([]float64, nClasses)
		for c := 0; c < nClasses; c++ {
			sum := 0.0
			for j := range model.CoefficientsList[c] {
				sum += XT[i][j] * model.CoefficientsList[c][j]
			}
			logits[i][c] = sum
		}
	}

	probs := softmax(logits)
	yPred := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		maxIdx := 0
		maxVal := probs[i][0]
		for j := 1; j < len(probs[i]); j++ {
			if probs[i][j] > maxVal {
				maxVal = probs[i][j]
				maxIdx = j
			}
		}
		yPred[i] = maxIdx
	}
	return yPred
}

func parseIndex(s string) int {
	var idx int
	fmt.Sscanf(s, "%d", &idx)
	return idx
}

func main() {
	file, err := os.ReadFile("outputs/iris_model.json")
	if err != nil {
		log.Fatal(err)
	}

	var model SymbolicModel
	if err := json.Unmarshal(file, &model); err != nil {
		log.Fatal(err)
	}

	X := [][]float64{
		{5.1, 3.5, 1.4, 0.2},
		{7.0, 3.2, 4.7, 1.4},
		{6.3, 3.3, 6.0, 2.5},
	}

	yPred := predict(X, model)
	fmt.Println("Predicted classes:", yPred)
}
