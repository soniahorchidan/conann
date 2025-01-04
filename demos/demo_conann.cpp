#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>

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
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // Distribution between 0 and 1
    for (size_t i = 0; i < 10000 * d; i++) {
        training_data[i] = dist(rng);  // Generate float between 0 and 1
    }
    index.train(10000, training_data.data());

    // generate random database
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = dist(rng);
    }

    { // populate the database
        index.add(nb, database.data());
    }

    size_t nq = 100;

    { // searching the database
        printf("Searching ...\n");

        std::vector<float> queries(nq * d);
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = dist(rng);
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

    // Test calibration
    auto lamhat = index.calibrate(0.1);
    std::cout << "Calibration done for alpha=0.1, lamhat= " << lamhat << std::endl;

    return 0;
}
