#pragma once
#include <vector>
#include <string>

class LSTM
{
public:
    LSTM(size_t input_size = 2, size_t hidden_size = 4, size_t output_size = 1);

    double forward(const std::vector<double>& input);
    void train(const std::vector<double>& input, double target, double lr = 0.01);
    void reset_state();
    bool save(const std::string& file) const;
    bool load(const std::string& file);

private:
    double sigmoid(double x) const;
    double dsigmoid(double y) const;
    double tanh_activate(double x) const;
    double dtanh(double y) const;

    size_t input_size_;
    size_t hidden_size_;
    size_t output_size_;

    std::vector<double> Wf, Uf, bf;
    std::vector<double> Wi, Ui, bi;
    std::vector<double> Wc, Uc, bc;
    std::vector<double> Wo, Uo, bo;
    std::vector<double> Wy, by;

    std::vector<double> h;
    std::vector<double> c;
};
