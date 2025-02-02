#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>


#include "NNLSRegression.h"



void NNLSRegression::fit(std::vector<std::vector<double>> x, std::vector<double> y, double learningRate, bool verbose) {
	int epochs = 1000000;
	this->weights.assign(x[0].size(), 0);
	this->alpha.assign(this->weights.size(), 0);
	std::vector<double> grad;
	for (int epoch = 0; epoch < epochs; epoch++) {
		grad = gradient(x, y);
		for (int j = 0; j < this->weights.size(); j++) {
			this->weights[j] -= grad[j] * learningRate;
		}
		for (int j = this->weights.size(); j < grad.size(); j++) {
			this->alpha[j - this->weights.size()] -= grad[j] * learningRate;
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



std::vector<double> NNLSRegression::gradient(std::vector<std::vector<double>> x, std::vector<double> y) {
	std::vector<double> grad;
	grad.assign(this->weights.size() + this -> alpha.size(), 0);
	for (int k = 0; k < this->weights.size(); k++) {
		double cur = 0;
		for (int n = 0; n < x.size(); n++) {
			double dotProduct = std::inner_product(this->weights.begin(), this->weights.end(), x[n].begin(), 0);
			cur += (y[n] - dotProduct) * x[n][k];
		}

		cur *= -2;
		cur -= this->alpha[k];
		grad[k] = cur;
	}
	for (int k = this->weights.size(); k < grad.size(); k++) {
		grad[k] -= this->weights[k - this->weights.size()];
	}
	return grad;
}


double NNLSRegression::loss(std::vector<std::vector<double>> x, std::vector<double> y) {
	double res = 0;
	for (int i = 0; i < x.size(); i++) {
		res += pow(std::inner_product(this->weights.begin(), this->weights.end(), x[i].begin(), 0) - y[i], 2);
	}
	return res - std::inner_product(this->weights.begin(), this->weights.end(), this->alpha.begin(), 0);
}



std::vector<double> NNLSRegression::predict(std::vector<std::vector<double>> x) {
	std::vector<double> predictions;
	for (int i = 0; i < x.size(); i++) {
		predictions.push_back(std::inner_product(this->weights.begin(), this->weights.end(), x[i].begin(), 0));
	}
	return predictions;
}