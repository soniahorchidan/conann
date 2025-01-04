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
    size_t nb = 10000;

    std::mt19937 rng(12345);

    // make the IVF index object and train it
    faiss::IndexFlatL2 quantizer(d);  // The quantizer (flat index)
    faiss::IndexIVFFlat index(&quantizer, d, 10, faiss::METRIC_L2);  // IVF index
    index.nprobe = 10;  // number of probes

    // train the index on some data
    std::vector<float> training_data(100 * d);  // Random training data
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // Distribution between 0 and 1
    for (size_t i = 0; i < 100 * d; i++) {
        training_data[i] = dist(rng);  // Generate float between 0 and 1
    }
    index.train(100, training_data.data());

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
    auto alpa = 0.1;
    auto lamhat = index.calibrate(alpa);
    auto fnr = index.evaluate_test(lamhat);
    std::cout << "alpha=" << alpa << ": lamhat= " << lamhat << ", test fnr=" << fnr << std::endl;

    alpa = 0.2;
    lamhat = index.calibrate(alpa);
    fnr = index.evaluate_test(lamhat);
    std::cout << "alpha=" << alpa << ": lamhat= " << lamhat << ", test fnr=" << fnr << std::endl;
    
    index.eval_on_lambda_range(0.1, 0.31, 0.1);

    return 0;
}
