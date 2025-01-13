# Structure:

./conann <- ConANN integrated into faiss1.9

./faiss <- faiss1.9

./LAET // optional

./Auncel <- ported to faiss 1.9


# How to build without GPU


Faiss:

```
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -B build .
make -C build -j faiss
```

ConANN:

```
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -B build .
make -C build -j demo_conann
```
