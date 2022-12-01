#include <stdio.h>
#include <time.h>

#define DEVICE 0
// Thread block size
#define BLOCK_SIZE 16
#define CUDA_FLOAT float

// Basic Matrix dimensions (can be amplified by command line switch)
// (chosen as multiples of the thread block size for simplicity)
#define WA (1  * BLOCK_SIZE) // Matrix A width
#define HA (1 * BLOCK_SIZE) // Matrix A height
#define WB (1  * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

// РџСЂРІРµСЂРєР° РЅР° РѕС€РёР±РєСѓ РІС‹РїРѕР»РЅРµРЅРёСЏ С„СѓРЅРєС†РёР№ РёР· cuda API
void check_cuda_error(const char *message)
{
	cudaError_t err = cudaGetLastError();
	if(err!=cudaSuccess){
		printf("ERROR: %s: %s\n", message, cudaGetErrorString(err) );
		exit(1);
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul( CUDA_FLOAT* C, CUDA_FLOAT* A, CUDA_FLOAT* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    CUDA_FLOAT Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
		//
		__shared__ CUDA_FLOAT At [BLOCK_SIZE][BLOCK_SIZE], //two temp matrices 
							  Bt [BLOCK_SIZE][BLOCK_SIZE]; //in shared memory
		
		At [tx][ty] = A [wB * ty + a + tx]; //per-element load 
		Bt [tx][ty] = B [wB * ty + b + tx]; //from global to shared
		
		__syncthreads(); //sync after loading
		
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += At[k][ty] * Bt[tx][k];
            //Csub += A[a + wA * ty + k] * B[b + wB * k + tx];
		__syncthreads(); 
		//
    }
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub; //write back to global
}

void randomInit(CUDA_FLOAT* data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (CUDA_FLOAT)RAND_MAX;
}

void computeGold(CUDA_FLOAT* C, const CUDA_FLOAT* A, const CUDA_FLOAT* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
  for (unsigned int i = 0; i < hA; ++i)
    for (unsigned int j = 0; j < wB; ++j) {
      double sum = 0;
      for (unsigned int k = 0; k < wA; ++k) {
        double a = A[i * wA + k];
        double b = B[k * wB + j];
        sum += a * b;
      }
      C[i * wB + j] = (CUDA_FLOAT)sum;
    }
}

bool IsEqual(CUDA_FLOAT *A, CUDA_FLOAT *B, unsigned int Size) {
  for(unsigned int i = 0; i < Size; i++)
    if(fabs(A[i] - B[i]) > 1e-4)
      return false;
  return true;
};

int main(int argc, char** argv)
{
  cudaSetDevice(DEVICE);	// Выбор устройства
  check_cuda_error("Error selecting device");
  CUDA_FLOAT *A_device, *B_device, *C_device;	//на устройстве
  CUDA_FLOAT A[WA * HA], B[WB * HB], C[WC * HC];	//в хостовой памяти
  cudaMalloc((void**)&A_device, sizeof(CUDA_FLOAT) * WA * HA);	// Выделение памяти на GPU
  check_cuda_error("Allocating memory on GPU");
  cudaMalloc((void**)&B_device, sizeof(CUDA_FLOAT) * WB * HB);	// Выделение памяти на GPU
  check_cuda_error("Allocating memory on GPU");
  cudaMalloc((void**)&C_device, sizeof(CUDA_FLOAT) * WC * HC);	// Выделение памяти на GPU
  check_cuda_error("Allocating memory on GPU");

  // Инициализируем мтрицы
  randomInit(A, WA * HA);
  randomInit(B, WB * HB);

  // Копируем на устройство
  cudaMemcpy(A_device, A, sizeof(CUDA_FLOAT) * WA * HA, cudaMemcpyHostToDevice);
  check_cuda_error("Copying data to GPU");
  cudaMemcpy(B_device, B, sizeof(CUDA_FLOAT) * WB * HB, cudaMemcpyHostToDevice);
  check_cuda_error("Copying data to GPU");

   // Рамеры грида и блока на GPU
   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
   dim3 grid(WC / block.x, HC / block.y);

	// Запускаем таймер
	time_t timer_rough;	// Ready
	clock_t timer_prec;	// Set
	timer_rough = time(0); timer_prec = clock();	// GO!

	matrixMul<<<grid, block>>>(C_device, A_device, B_device, WA, WB);	// Запуск ядра
	cudaThreadSynchronize();	// Ожидаем завершения работы ядра
	check_cuda_error("Executing kernel");

	// Останавливаем таймер
	timer_rough = time(0) - timer_rough; timer_prec = clock() - timer_prec;
	printf("Kernel time: ");
	if (timer_rough > 100)
		printf("%d (s)\n",(int)timer_rough);
	else
		printf("%f (s)\n",timer_prec * 1.0 / CLOCKS_PER_SEC);

	cudaMemcpy(C, C_device, sizeof(CUDA_FLOAT) * WC * HC, cudaMemcpyDeviceToHost);	// Копируем результаты на хост
	check_cuda_error("Copying results from GPU");
	cudaFree(A_device);	// Освобождаем память на GPU
	check_cuda_error("Freeing device memory");
	cudaFree(B_device);	// Освобождаем память на GPU
	check_cuda_error("Freeing device memory");
	cudaFree(C_device);	// Освобождаем память на GPU
	check_cuda_error("Freeing device memory");

  for (int x = 0; x < WC; x++) {
    for (int y = 0; y < HC; y++) 
      printf("%f ", C[y * WC + x]);
    printf("\n");
  }

  CUDA_FLOAT C_check[WC * HC];
  computeGold(C_check, A, B, HA, WA, WB);
  if(IsEqual(C, C_check, WC*HC))
    printf("Correct.\n");
  else {
    printf("InCorrect!\n");
    for (int x = 0; x < WC; x++) {
      for (int y = 0; y < HC; y++) 
        printf("%f ", C_check[y * WC + x]);
      printf("\n");
   }
  }
  return 0;
}
