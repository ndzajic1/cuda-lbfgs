#include <chrono>
#include <random>
#include <benchmark.h>
#include <matrices.h>
#include <vector_utils.h>

double quadratic(const vector<double> &X) {
    double sum = 0.0;
    for (const double x : X) {
        sum += (x - 1) * (x - 1);
    }
    return sum;
}

vector<double> quadratic_grad(const vector<double> &X) {
    vector<double> grad(X.size());
    for (size_t i = 0; i < X.size(); i++) {
        grad[i] = 2.0 * (X[i] - 1);
    }
    return grad;
}


int main()
{   

    // dimension
    int dim = 10000;

  //  auto quadratic = generate_quadratic_function(dim);
   // auto quadratic_grad = generate_quadratic_gradient(dim);

    // random seeds 42, 365, 12345, 777777, 10000
    unsigned seed = 42;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1000, 1000);
    std::vector<double> x0(dim);

    for (double &num : x0)
        {
            num = dis(gen);
        }  

    double res = benchmark(
            "Quadratic Function",
            quadratic,
            quadratic_grad,
            x0,    // x0
            15000, // max_iterations
            10,    // m
            1e-8   // tolerance
        );

    cout << "Elapsed time: " << res << " seconds." << endl;

    return 0;
}
