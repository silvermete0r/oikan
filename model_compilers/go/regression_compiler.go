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
	BasisFunctions []string  `json:"basis_functions"`
	NFeatures      int       `json:"n_features"`
	Coefficients   []float64 `json:"coefficients"`
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
				val = math.Exp(math.Min(math.Max(X[i][idx], -10), 10))
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

func predict(X [][]float64, model SymbolicModel) []float64 {
	XT := evaluateBasisFunctions(X, model.BasisFunctions, model.NFeatures)
	yPred := make([]float64, len(X))
	for i := range X {
		sum := 0.0
		for j, coef := range model.Coefficients {
			sum += XT[i][j] * coef
		}
		yPred[i] = sum
	}
	return yPred
}

func parseIndex(s string) int {
	var idx int
	fmt.Sscanf(s, "%d", &idx)
	return idx
}

func main() {
	data, err := os.ReadFile("outputs/california_housing_model.json")
	if err != nil {
		log.Fatal(err)
	}

	var model SymbolicModel
	if err := json.Unmarshal(data, &model); err != nil {
		log.Fatal(err)
	}

	// Generate random X for testing (10 samples)
	X := make([][]float64, 10)
	for i := range X {
		X[i] = make([]float64, model.NFeatures)
		for j := 0; j < model.NFeatures; j++ {
			X[i][j] = randFloat64()
		}
	}

	yPred := predict(X, model)
	fmt.Println("Predictions:", yPred)
}

func randFloat64() float64 {
	return float64(randInt(0, 10000)) / 10000.0
}

func randInt(min int, max int) int {
	return min + int(math.Floor(float64(max-min+1)*randNorm()))
}

func randNorm() float64 {
	return math.Abs(math.Sin(float64(os.Getpid()) + float64(os.Getppid())))
}
