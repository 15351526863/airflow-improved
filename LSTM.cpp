#include "LSTM.hpp"
#include <fstream>
#include <random>
#include <cmath>
#include <filesystem>

namespace {
    double dot(const std::vector<double>& a, const std::vector<double>& b, size_t rows, size_t cols, size_t row) {
        double r = 0.0;
        for (size_t i = 0; i < cols; ++i)
            r += a[row * cols + i] * b[i];
        return r;
    }
}

LSTM::LSTM(size_t input_size, size_t hidden_size, size_t output_size)
    : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size),
      Wf(hidden_size * input_size), Uf(hidden_size * hidden_size), bf(hidden_size),
      Wi(hidden_size * input_size), Ui(hidden_size * hidden_size), bi(hidden_size),
      Wc(hidden_size * input_size), Uc(hidden_size * hidden_size), bc(hidden_size),
      Wo(hidden_size * input_size), Uo(hidden_size * hidden_size), bo(hidden_size),
      Wy(output_size * hidden_size), by(output_size), h(hidden_size, 0.0), c(hidden_size, 0.0)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    auto init = [&](std::vector<double>& v) { for (double& d : v) d = dist(rng); };
    init(Wf); init(Uf); init(bf);
    init(Wi); init(Ui); init(bi);
    init(Wc); init(Uc); init(bc);
    init(Wo); init(Uo); init(bo);
    init(Wy); init(by);
}

double LSTM::sigmoid(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
double LSTM::dsigmoid(double y) const { return y * (1.0 - y); }
double LSTM::tanh_activate(double x) const { return std::tanh(x); }
double LSTM::dtanh(double y) const { return 1.0 - y * y; }

double LSTM::forward(const std::vector<double>& input)
{
    std::vector<double> f(hidden_size_);
    std::vector<double> i_g(hidden_size_);
    std::vector<double> g(hidden_size_);
    std::vector<double> o(hidden_size_);
    for (size_t h_idx = 0; h_idx < hidden_size_; ++h_idx) {
        double xf = dot(Wf, input, hidden_size_, input_size_, h_idx);
        double hf = dot(Uf, h, hidden_size_, hidden_size_, h_idx);
        f[h_idx] = sigmoid(xf + hf + bf[h_idx]);

        double xi = dot(Wi, input, hidden_size_, input_size_, h_idx);
        double hi = dot(Ui, h, hidden_size_, hidden_size_, h_idx);
        i_g[h_idx] = sigmoid(xi + hi + bi[h_idx]);

        double xc = dot(Wc, input, hidden_size_, input_size_, h_idx);
        double hc = dot(Uc, h, hidden_size_, hidden_size_, h_idx);
        g[h_idx] = tanh_activate(xc + hc + bc[h_idx]);

        double xo = dot(Wo, input, hidden_size_, input_size_, h_idx);
        double ho = dot(Uo, h, hidden_size_, hidden_size_, h_idx);
        o[h_idx] = sigmoid(xo + ho + bo[h_idx]);

        c[h_idx] = f[h_idx] * c[h_idx] + i_g[h_idx] * g[h_idx];
        h[h_idx] = o[h_idx] * tanh_activate(c[h_idx]);
    }

    double v = dot(Wy, h, output_size_, hidden_size_, 0) + by[0];
    return sigmoid(v);
}

void LSTM::train(const std::vector<double>& input, double target, double lr)
{
    std::vector<double> h_prev = h;
    std::vector<double> c_prev = c;

    std::vector<double> f(hidden_size_), i_g(hidden_size_), g(hidden_size_), o(hidden_size_);
    for (size_t h_idx = 0; h_idx < hidden_size_; ++h_idx) {
        double xf = dot(Wf, input, hidden_size_, input_size_, h_idx);
        double hf = dot(Uf, h_prev, hidden_size_, hidden_size_, h_idx);
        f[h_idx] = sigmoid(xf + hf + bf[h_idx]);

        double xi = dot(Wi, input, hidden_size_, input_size_, h_idx);
        double hi = dot(Ui, h_prev, hidden_size_, hidden_size_, h_idx);
        i_g[h_idx] = sigmoid(xi + hi + bi[h_idx]);

        double xc = dot(Wc, input, hidden_size_, input_size_, h_idx);
        double hc = dot(Uc, h_prev, hidden_size_, hidden_size_, h_idx);
        g[h_idx] = tanh_activate(xc + hc + bc[h_idx]);

        double xo = dot(Wo, input, hidden_size_, input_size_, h_idx);
        double ho = dot(Uo, h_prev, hidden_size_, hidden_size_, h_idx);
        o[h_idx] = sigmoid(xo + ho + bo[h_idx]);

        c[h_idx] = f[h_idx] * c_prev[h_idx] + i_g[h_idx] * g[h_idx];
        h[h_idx] = o[h_idx] * tanh_activate(c[h_idx]);
    }

    double y = sigmoid(dot(Wy, h, output_size_, hidden_size_, 0) + by[0]);
    double dy = (y - target) * dsigmoid(y);
    for (size_t k = 0; k < hidden_size_; ++k)
        Wy[k] -= lr * dy * h[k];
    by[0] -= lr * dy;

    std::vector<double> dh(hidden_size_, dy * Wy[0]);
    std::vector<double> dc(hidden_size_, 0.0);

    for (size_t h_idx = 0; h_idx < hidden_size_; ++h_idx) {
        double tanhc = tanh_activate(c[h_idx]);
        double do_ = dh[h_idx] * tanhc * dsigmoid(o[h_idx]);
        double dc_new = dh[h_idx] * o[h_idx] * dtanh(tanhc) + dc[h_idx];

        double df = dc_new * c_prev[h_idx] * dsigmoid(f[h_idx]);
        double di = dc_new * g[h_idx] * dsigmoid(i_g[h_idx]);
        double dg = dc_new * i_g[h_idx] * dtanh(g[h_idx]);

        for (size_t p = 0; p < input_size_; ++p) {
            Wf[h_idx * input_size_ + p] -= lr * df * input[p];
            Wi[h_idx * input_size_ + p] -= lr * di * input[p];
            Wc[h_idx * input_size_ + p] -= lr * dg * input[p];
            Wo[h_idx * input_size_ + p] -= lr * do_ * input[p];
        }

        for (size_t p = 0; p < hidden_size_; ++p) {
            Uf[h_idx * hidden_size_ + p] -= lr * df * h_prev[p];
            Ui[h_idx * hidden_size_ + p] -= lr * di * h_prev[p];
            Uc[h_idx * hidden_size_ + p] -= lr * dg * h_prev[p];
            Uo[h_idx * hidden_size_ + p] -= lr * do_ * h_prev[p];
        }

        bf[h_idx] -= lr * df;
        bi[h_idx] -= lr * di;
        bc[h_idx] -= lr * dg;
        bo[h_idx] -= lr * do_;

        dc[h_idx] = dc_new * f[h_idx];
    }
}

void LSTM::reset_state()
{
    std::fill(h.begin(), h.end(), 0.0);
    std::fill(c.begin(), c.end(), 0.0);
}

bool LSTM::save(const std::string& file) const
{
    std::ofstream out(file, std::ios::binary);
    if (!out)
        return false;
    auto write = [&](const std::vector<double>& v) { out.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double)); };
    write(Wf); write(Uf); write(bf);
    write(Wi); write(Ui); write(bi);
    write(Wc); write(Uc); write(bc);
    write(Wo); write(Uo); write(bo);
    write(Wy); write(by);
    return true;
}

bool LSTM::load(const std::string& file)
{
    std::ifstream in(file, std::ios::binary);
    if (!in)
        return false;
    auto read = [&](std::vector<double>& v) { in.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(double)); };
    read(Wf); read(Uf); read(bf);
    read(Wi); read(Ui); read(bi);
    read(Wc); read(Uc); read(bc);
    read(Wo); read(Uo); read(bo);
    read(Wy); read(by);
    return true;
}

