#pragma once
#ifndef RIDGEREGRESSION_H
#define RIDGEREGRESSION_H
#include <vector>




class RidgeRegression {
private:
	std::vector<double> weights;
	double alpha = 0;
	std::vector<double> gradient(std::vector<std::vector<double>> x, std::vector<double> y);
	double loss(std::vector < std::vector < double>> x, std::vector < double> y);
public:
	std::vector<double> coefficients = std::vector<double>(0);
	double intercept = 0;
	void fit(std::vector<std::vector<double>> x, std::vector<double> y, double learningRate, bool verbose);
	std::vector<double> predict(std::vector<std::vector<double>> x);
	RidgeRegression(double alpha);
};




#endif