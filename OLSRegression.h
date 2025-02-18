#pragma once
#ifndef OLSREGRESSION_H
#define OLSREGRESSION_H
#include <vector>




class OLSRegression {
private:
	std::vector<std::vector<double>> x;
	std::vector < double> y;
	std::vector<double> weights;
	std::vector<double> gradient(std::vector<std::vector<double>> x, std::vector<double> y);
	double loss(std::vector < std::vector < double>> x, std::vector < double> y);
public:
	std::vector<double> coefficients = std::vector<double>(0);
	double intercept = 0;
	void fit(std::vector<std::vector<double>> x, std::vector<double> y, double learningRate, bool verbose);
	std::vector<double> predict(std::vector<std::vector<double>> x);

};




#endif