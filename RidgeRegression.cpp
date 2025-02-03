#include <iostream>
#include <math.h>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "RidgeRegression.h"



RidgeRegression::RidgeRegression(double alpha) {
	if (alpha < 0) {
		throw std::invalid_argument("Alpha parameter has to be positive");
	}
	this->alpha = alpha;
}



void RidgeRegression::fit(std::vector<std::vector<double>> x, std::vector<double> y, double learningRate, bool verbose) {
	int epochs = 1000000;
	this->weights.assign(x[0].size(), 0);
	std::vector<double> grad;
	for (int epoch = 0; epoch < epochs; epoch++) {
		grad = gradient(x, y);
		for (int j = 0; j < this->weights.size(); j++) {
			this->weights[j] -= grad[j] * learningRate;
		}
		if (verbose && epoch % 100 == 0) {

			std::cout << "Loss: " << this->loss(x, y) << std::endl;
		}

	}
	this->coefficients = std::vector<double>();
	for (int i = 0; i < this->weights.size() - 1; i++) {
		this->coefficients.push_back(weights[i]);
	}
	this->intercept = this->weights[this->weights.size() - 1];
}

std::vector<double> RidgeRegression::gradient(std::vector<std::vector<double>> x, std::vector<double> y) {
	std::vector<double> grad;
	grad.assign(this->weights.size(), 0);
	for (int k = 0; k < grad.size(); k++) {
		double cur = 0;
		for (int n = 0; n < x.size(); n++) {
			double dotProduct = std::inner_product(this->weights.begin(), this->weights.end(), x[n].begin(), 0);
			cur += (y[n] - dotProduct) * x[n][k];
		}

		cur *= -2;
		cur += 2 * this->alpha * this -> weights[k];
		grad[k] = cur;
	}
	return grad;
}


double RidgeRegression::loss(std::vector<std::vector<double>> x, std::vector<double> y) {
	double res = 0;
	for (int i = 0; i < x.size(); i++) {
		res += pow(std::inner_product(this->weights.begin(), this->weights.end(), x[i].begin(), 0) - y[i], 2);
	}
	return res + this->alpha * std::inner_product(this->weights.begin(), this->weights.end(), this->weights.begin(), 0) ;
}



std::vector<double> RidgeRegression::predict(std::vector<std::vector<double>> x) {
	std::vector<double> predictions;
	for (int i = 0; i < x.size(); i++) {
		predictions.push_back(std::inner_product(this->weights.begin(), this->weights.end(), x[i].begin(), 0));
	}
	return predictions;
}