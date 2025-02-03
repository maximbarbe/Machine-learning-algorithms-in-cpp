#pragma once
#ifndef LASSOREGRESSION_H
#define LASSOREGRESSION_H
#include <vector>




class LassoRegression {
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
	LassoRegression(double alpha);
};




#endif