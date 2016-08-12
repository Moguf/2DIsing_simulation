Matirx Size [3072,3072]
=== nvprof ===
grid=[64,64],block[32,32]
==13650== NVPROF is profiling process 13650, command: ./run
==13650== Profiling application: ./run
==13650== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.86%  12.4234s       200  62.117ms  55.989ms  62.822ms  g_simulate(int, float, int*, int*, int, int, curandStateXORWOW*, int)
  0.06%  7.3020ms         1  7.3020ms  7.3020ms  7.3020ms  g_S_init(int*, int, curandStateXORWOW*)
  0.05%  6.3272ms         1  6.3272ms  6.3272ms  6.3272ms  g_rand_init(int, curandStateXORWOW*)
  0.03%  4.0466ms         1  4.0466ms  4.0466ms  4.0466ms  [CUDA memcpy DtoH]

==13650== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 98.79%  12.4490s         3  4.14965s  74.521us  12.4488s  cudaFree
  0.76%  95.638ms         3  31.880ms  135.49us  95.320ms  cudaMalloc
  0.28%  35.886ms         1  35.886ms  35.886ms  35.886ms  cudaDeviceReset
  0.14%  17.797ms         1  17.797ms  17.797ms  17.797ms  cudaMemcpy
  0.01%  1.0409ms      1605     648ns     628ns  6.9150us  cudaSetupArgument
  0.01%  1.0277ms       202  5.0870us  4.5390us  41.277us  cudaLaunch
  0.00%  383.51us        83  4.6200us     768ns  145.06us  cuDeviceGetAttribute
  0.00%  135.71us       202     671ns     628ns  2.6540us  cudaConfigureCall
  0.00%  46.445us         1  46.445us  46.445us  46.445us  cuDeviceTotalMem
  0.00%  38.553us         1  38.553us  38.553us  38.553us  cuDeviceGetName
  0.00%  2.8630us         2  1.4310us     908ns  1.9550us  cuDeviceGetCount
  0.00%  1.7460us         2     873ns     768ns     978ns  cuDeviceGet
=== time command ===
real	0m13.601s
user	0m3.064s
sys	0m10.452s