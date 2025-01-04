#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>

#include <gsl/gsl_sf_bessel.h>

using namespace std::chrono;

int main(void) {
    // dimension of the vectors to index
    int d = 32;
    int K = 64;

    // size of the database we plan to index
    size_t nb = 1000;

    std::mt19937 rng(12345);

    // make the IVF index object and train it
    faiss::IndexFlatL2 quantizer(d);  // The quantizer (flat index)
    faiss::IndexIVFFlat index(&quantizer, d, 10, faiss::METRIC_L2);  // IVF index
    index.nprobe = 10;  // number of probes

    // train the index on some data
    std::vector<float> training_data(10000 * d);  // Random training data
    for (size_t i = 0; i < 10000 * d; i++) {
        training_data[i] = rng() % 1024;
    }
    index.train(10000, training_data.data());

    // generate random database
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = rng() % 1024;
    }

    { // populate the database
        index.add(nb, database.data());
    }

    size_t nq = 100;

    { // searching the database
        printf("Searching ...\n");

        std::vector<float> queries(nq * d);
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = rng() % 1024;
        }

        int k = 5;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        // Output results
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1000 * 1000 / t;

        printf("QPS: %d\n", qps);
    }

    // Example prediction sets and ground truth labels
    std::vector<std::vector<int>> prediction_set = {{0, 1, 2}, {3, 4}, {5, 6}};
    std::vector<std::vector<int>> gt_labels = {{0, 1, 3}, {3, 5}, {6, 7}};
    float fnr = index.false_negative_rate(prediction_set, gt_labels);
    std::cout << "False Negative Rate Example: " << fnr << std::endl;

    // GSL test
    double x = 15.0;
    double y = gsl_sf_bessel_J0 (x);
    printf ("J0(%g) = %.18e\n", x, y);

    return 0;
}
