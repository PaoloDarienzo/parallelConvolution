#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include <sys/time.h> // for Linux

//#include <iostream>
//#include <windows.h> // for Windows APIs
//using namespace std;

/*!
* \brief Convolution on a 2D array, with a 2D mask. Tiled, with constant memory, with shared memory.
* It's a more readable version of the convolutionConstantTiled program.
* \author Paolo D'Arienzo
*/

#define MASK_WIDTH1 77 //kernel rows
#define MASK_WIDTH2 77 //kernel columns
#define WIDTH1 1000 //rows
#define WIDTH2 1000 //columns
#define TILE_SIZE 32

#define DIM1 (TILE_SIZE + (MASK_WIDTH1 - 1)) //first dimension of N_ds
#define DIM2 (TILE_SIZE + (MASK_WIDTH2 - 1)) //second dimension of N_ds
#define XRADIUS (int)(MASK_WIDTH1 / 2)
#define YRADIUS (int)(MASK_WIDTH2 / 2)

/*!
*Declaring constant matrix M of dimension MASK_WIDTH1xMASK_WIDTH2
*/
__constant__ float M[MASK_WIDTH1][MASK_WIDTH2];


__host__ void matrix_Print(float A[][WIDTH2]);

__host__ void matrix_Print_Mask(float h_M[][MASK_WIDTH2]);

__host__ int mask_Loading(float M[][MASK_WIDTH2]);

__host__ int input_Loading(float N[][WIDTH2]);

__host__ int output_Writing(float P[][WIDTH2]);

__host__ int convolution_2D(float N[][WIDTH2], float h_M[][MASK_WIDTH2], float P[][WIDTH2]);

__global__ void convolution_2D_Kernel(float *N, float *P);


/*!
* Host function that launches the kernel function of convolution.
* @param N, M, P pointers to input and output matrices.
*/
__host__ int convolution_2D(float N[][WIDTH2], float h_M[][MASK_WIDTH2], float P[][WIDTH2]) {

	/*
	//Time calculation on Windows
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	QueryPerformanceFrequency(&frequency);
	*/

	//Time calculation on Linux
	struct timeval  tv1, tv2;

	double timeExec; //contains time of execution

	FILE *fp;

	//Pointer to the file where will be stored the execution time
	fp = fopen("/storage/pdarienzo/tirocinio/tesi/timestat.txt", "a");
	if (fp == NULL) {
		printf("Failed to open timestat.txt file.\n");
		return(-1);
	}

	int IOSize = WIDTH1 * WIDTH2 * sizeof(float);
	int kernelSize = MASK_WIDTH1 * MASK_WIDTH2 * sizeof(float);
	float *dN, *dP;

	//Allocating pointers in global memory of device
	cudaMalloc((void**)&dN, IOSize);
	cudaMemcpy(dN, N, IOSize, cudaMemcpyHostToDevice);

	//Copying mask matrix in constant memory
	cudaMemcpyToSymbol(M, h_M, kernelSize);

	cudaMalloc((void**)&dP, IOSize);
	cudaMemcpy(dP, P, IOSize, cudaMemcpyHostToDevice); //P will be totally overwritten, copy not necessary

	//Each block has the dimension of a tile
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((WIDTH2 - 1) / TILE_SIZE + 1, (WIDTH1 - 1) / TILE_SIZE + 1, 1);

	printf("Convolution, grid dimension %ix%i; thread blocks dimension: %ix%i...\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

	// start timer
	//Time calculation on Linux
	gettimeofday(&tv1, NULL);
	//Time calculation on Windows
	//QueryPerformanceCounter(&t1);

	//Convolution
	convolution_2D_Kernel << <dimGrid, dimBlock >> > (dN, dP);

	// stop timer
	//Time calculation on Linux
	gettimeofday(&tv2, NULL);
	//Time calculation on Windows
	//QueryPerformanceCounter(&t2);

	//Copying results from device in P
	cudaMemcpy(P, dP, IOSize, cudaMemcpyDeviceToHost);

	//Pointers free
	cudaFree(dN);
	cudaFree(dP);

	/*
	//Time calculation on Windows
	// compute and print the elapsed time in millisec
	timeExec = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << elapsedTime << " ms.\n";
	*/
	//Time calculation on Linux
	timeExec = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
	printf("Total time = %lf seconds\n", timeExec);

	fprintf(fp, "Execution time of (optimized) parallel convolution 2D constant tiled: %lf seconds.", timeExec);
	fprintf(fp, "\n");

	fclose(fp);

	return 0;

}

/*!
* Device function that implements the loading of ghost elements in N_ds.
* @param N_ds pointer to matrix in shared memory
* @param p_coord_x, p_coord_y pointers to N_ds coordinates
* @param p_my_first_elem_x, p_my_first_elem_y pointers to variables that host coordinates of N elements that have to be loaded
* @param p_first_thread_in_block_x, p_first_thread_in_block_y pointers to shared variables 
* that host coordinates of the first N element that is loaded by the first thread in the block thread
*/
__device__ void ghost_Loading(float N_ds[][DIM2], int *p_coord_x, int *p_coord_y, int *p_my_first_elem_x, int *p_my_first_elem_y,
	int *p_first_thread_in_block_x, int *p_first_thread_in_block_y) {

	if ((*p_coord_y) == (DIM2 - 1)) { //Reached last element of row in N_ds
		N_ds[(*p_coord_x)][(*p_coord_y)] = 0;

		if ((*p_coord_y) % (DIM2 - 1) == 0 && (*p_coord_y) != 0) {
			(*p_my_first_elem_y) = (*p_first_thread_in_block_y) - YRADIUS;
			(*p_my_first_elem_x)++;
		}
		else {
			(*p_my_first_elem_y)++;
		}

		(*p_coord_y) = 0;
		(*p_coord_x)++;
	}
	else {
		N_ds[(*p_coord_x)][(*p_coord_y)] = 0;

		if ((*p_coord_y) % (DIM2 - 1) == 0 && (*p_coord_y) != 0) {
			(*p_my_first_elem_y) = (*p_first_thread_in_block_y) - YRADIUS;
			(*p_my_first_elem_x)++;
		}
		else {
			(*p_my_first_elem_y)++;
		}

		(*p_coord_y)++;
	}



}

/*!
* Device function that implements the loading of N elements in N_ds.
* @param N pointer to input matrix N
* @param N_ds pointer to matrix in shared memory
* @param p_coord_x, p_coord_y pointers to N_ds coordinates
* @param p_my_first_elem_x, p_my_first_elem_y pointers to variables that host coordinates of N elements that have to be loaded
* @param p_first_thread_in_block_x, p_first_thread_in_block_y pointers to shared variables
* that host coordinates of the first N element that is loaded by the first thread in the block thread
*/
__device__ void input_Copying(float *N, float N_ds[][DIM2], int *p_coord_x, int *p_coord_y, int *p_my_first_elem_x, int *p_my_first_elem_y,
	int *p_first_thread_in_block_x, int *p_first_thread_in_block_y) {

	if ((*p_coord_y) == (DIM2 - 1)) { //Reached last element of row in N_ds
		N_ds[(*p_coord_x)][(*p_coord_y)] = N[((*p_my_first_elem_x) * WIDTH2 + (*p_my_first_elem_y))];

		if ((*p_coord_y) % (DIM2 - 1) == 0 && (*p_coord_y) != 0) {
			(*p_my_first_elem_y) = (*p_first_thread_in_block_y) - YRADIUS;
			(*p_my_first_elem_x)++;
		}
		else {
			(*p_my_first_elem_y)++;
		}

		(*p_coord_y) = 0;
		(*p_coord_x)++;
	}
	else {

		N_ds[(*p_coord_x)][(*p_coord_y)] = N[((*p_my_first_elem_x) * WIDTH2 + (*p_my_first_elem_y))];

		if ((*p_coord_y) % (DIM2 - 1) == 0 && (*p_coord_y) != 0) {
			(*p_my_first_elem_y) = (*p_first_thread_in_block_y) - YRADIUS;
			(*p_my_first_elem_x)++;
		}
		else {
			(*p_my_first_elem_y)++;
		}

		(*p_coord_y)++;
	}

}

/*!
* Kernel function of convolution.
* @param N, P pointers to input and output matrices.
*/
__global__ void convolution_2D_Kernel(float *N, float *P) {

	//Declaring N_ds
	__shared__ float N_ds[DIM1][DIM2];

	//Declaring variable that will host the x value of the first thread of each block
	__shared__ int first_thread_in_block_x;
	//Declaring variable that will host the y value of the first thread of each block
	__shared__ int first_thread_in_block_y;

	int g_tx = blockIdx.x*blockDim.x + threadIdx.x; //x coord of thread in grid
	int g_ty = blockIdx.y*blockDim.y + threadIdx.y; //y coord of thread in grid

	//Needed only for output debugging
	//int blockId = blockIdx.x + blockIdx.y * gridDim.x; //id blocks
	//int g_i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; //id global of thread

	int tx = threadIdx.x; //x coord of thread in block
	int ty = threadIdx.y; //y coord of thread in block
	int b_i = ty * blockDim.x + tx; //id of the thread in block

	//Setting in the shared variables the coordinates of the first thread in block
	if (b_i == 0) {
		first_thread_in_block_x = g_ty;
		first_thread_in_block_y = g_tx;
	}
	__syncthreads(); //Waiting the writing in the shared variables

	//Number of elements each threads will load: values to load divided by the number of threads
	int elements_to_load = (DIM1 * DIM2) / (TILE_SIZE * TILE_SIZE) + 1; //e.g. 25/9 + 1 = 3
	//NB: elements_to_load performs an integer division, therefore 1 value could be lost

	//Number of total elements divided by the number of elements each threads will load, 
	//gives the number of threads engaged in loading
	int loading_threads = (DIM1 * DIM2) / elements_to_load + 1;

	//Temporary output value
	float PValue = 0;

	//Threads engaged in the loading are the first "elements_to_load" threads
	if (b_i < loading_threads) {

		//th0 starts from 0, th1 from 3, th2 from 6 and so on (if elements_to_load is 3)
		//First element the thread should load
		int first_element_to_load = b_i * elements_to_load;

		//Converting the value in matrix coordinates and reversing N_ds matrix (that is included in computation of first element)
		int coord_x = first_element_to_load / DIM2; //coord_x of first element the thread have to load (in N_ds)
		int coord_y = first_element_to_load % DIM2; //coord_y of first element the thread have to load (in N_ds)

		int my_first_elem_x = first_thread_in_block_x - XRADIUS;
		int my_first_elem_y = first_thread_in_block_y - YRADIUS;

		int *p_coord_x = &coord_x;
		int *p_coord_y = &coord_y;
		int *p_my_first_elem_x = &my_first_elem_x;
		int *p_my_first_elem_y = &my_first_elem_y;
		int *p_first_thread_in_block_x = &first_thread_in_block_x;
		int *p_first_thread_in_block_y = &first_thread_in_block_y;

		//Updating my_first_elem_x and my_first_elem_y
		for (int i = 0; i < first_element_to_load; i++) {
			if ((i + 1) % DIM2 == 0 && (i + 1) != 0) {
				my_first_elem_y = first_thread_in_block_y - YRADIUS;
				my_first_elem_x++;
			}
			else
				my_first_elem_y++;
		}

		//Number of ghost elements beyond the right border
		int ghost_over_dx = (DIM2 - (WIDTH2 % TILE_SIZE)) - YRADIUS;
		//int ghost_over_sx = YRADIUS; //Number of ghost elements before the left border

		//Declaring boolean that denote if the block in which the thread resides is on the left border (upper)
		bool sxU = false;
		//Declaring boolean that denote if the block in which the thread resides is on the right border (upper)
		bool dxU = false;
		//Declaring boolean that denote if the block in which the thread resides is on the left border (lower)
		bool sxL = false;
		//Declaring boolean that denote if the block in which the thread resides is on the right border (lower)
		bool dxL = false;
		//Declaring boolean that denote if the block in which the thread resides is on the left border (central)
		bool sxC = false;
		//Declaring boolean that denote if the block in which the thread resides is on the right border (central)
		bool dxC = false;
		//Declaring boolean that denote if the block in which the thread resides is central (upper)
		bool upC = false;
		//Declaring boolean that denote if the block in which the thread resides is central (lower)
		bool dwnC = false;
		//Declaring boolean that denote if the block in which the thread resides is just not on border, here called central
		bool C = false; //Not used

		//Setting all booleans
		if (blockIdx.x == 0) { //left side
			if (blockIdx.y == 0) { //upper
				sxU = true;
			}
			else if (blockIdx.y == (gridDim.y - 1)) { //lower
				sxL = true;
			}
			else //it's just central (on the left)
				sxC = true;
		}
		else if (blockIdx.x == (gridDim.x - 1)) {
			if (blockIdx.y == 0) { //upper
				dxU = true;
			}
			else if (blockIdx.y == (gridDim.y - 1)) { //lower
				dxL = true;
			}
			else //it's just central (on the right)
				dxC = true;
		}
		else if (blockIdx.y == 0) { //central in the upper side
			upC = true;
		}
		else if (blockIdx.y == (gridDim.y - 1)) { //central in the lower side
			dwnC = true;
		}
		else { //the remaining blocks are central
			   //Never used
			C = true;
		}

		for (int z = 0; z < elements_to_load; z++) { //Counting inserted elements with z

			if (sxU) {

				if ((coord_x < XRADIUS) ||
					(coord_y < YRADIUS)) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			}//end sxU

			else if (sxL) {

				if ((coord_y < YRADIUS) ||
					(coord_x >= ((WIDTH1 % TILE_SIZE) + XRADIUS))) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			}//end sxL

			else if (sxC) {

				if (coord_y < YRADIUS) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			}//end sxC

			else if (dxU) {

				if ((coord_x < XRADIUS) ||
					(coord_y >= (DIM2 - ghost_over_dx))) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			}//end dxU

			else if (dxL) {

				if ((coord_y >= (DIM2 - ghost_over_dx)) ||
					(coord_x >= ((WIDTH1 % TILE_SIZE) + XRADIUS))) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			}//end dxL

			else if (dxC) {

				if (coord_y >= (DIM2 - ghost_over_dx)) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			}//end dxC

			else if (upC) {

				if (coord_x < XRADIUS) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			} //end upC

			else if (dwnC) {

				if (coord_x >= ((WIDTH1 % TILE_SIZE) + XRADIUS)) { //Ghost elements

					ghost_Loading(N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				} //end ghost elem

				else {

					input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

				}

			} //end dwnC

			else {

				input_Copying(N, N_ds, p_coord_x, p_coord_y, p_my_first_elem_x, p_my_first_elem_y, p_first_thread_in_block_x, p_first_thread_in_block_y);

			}
		}//End of for cycle
	}//End of threads responsible for loading

	 //Waiting the loading in N_ds
	__syncthreads();

	//Only threads that cover P work on output
	//!(g_tx >= WIDTH2 || g_ty >= WIDTH1)
	if (g_tx < WIDTH2 && g_ty < WIDTH1) {

		//Convolution
		for (int i = 0; i < MASK_WIDTH1; i++) {
			for (int j = 0; j < MASK_WIDTH2; j++) {
				PValue += N_ds[ty + i][tx + j] * M[i][j];
			}
		}

		//shrewdness used by the book
		__syncthreads();

		//Loading in P
		P[g_tx + g_ty*WIDTH2] = PValue;

	}

}

int main() {

	float N[WIDTH1][WIDTH2] = { { 0 } };

	float P[WIDTH1][WIDTH2] = { 0 };

	float h_M[MASK_WIDTH1][MASK_WIDTH2] = { { 0 } };

	printf("Loading input matrix N... \n");
	input_Loading(N);
	printf("N loaded. \n");
	//matrix_Print(N);

	printf("Loading kernel matrix M... \n");
	mask_Loading(h_M);
	printf("h_M loaded. \n");
	//matrix_Print_Mask(h_M);

	printf("Convolution... \n");

	//Convolution
	convolution_2D(N, h_M, P);

	printf("Writing resulting matrix P. \n");
	//matrix_Print(P);
	output_Writing(P);

	printf("Press any key to continue...");
	//getc(stdin);

	return 0;
}

/*!
* Host function that prints the input/output matrix.
* @param N pointer to input matrix.
*/
__host__ void matrix_Print(float A[][WIDTH2]) {

	for (int i = 0; i < WIDTH1; i++) {
		for (int j = 0; j < WIDTH2; j++)
			printf("%f\t", A[i][j]);
		printf("\n");
	}
	printf("\n");

}

/*!
* Host function that prints the mask matrix.
* @param M pointer to mask matrix.
*/
__host__ void matrix_Print_Mask(float M[][MASK_WIDTH2]) {

	for (int i = 0; i < MASK_WIDTH1; i++) {
		for (int j = 0; j < MASK_WIDTH2; j++)
			printf("%f\t", M[i][j]);
		printf("\n");
	}
	printf("\n");

}

/*!
* Host function of input loading that reads from a file.
* @param N pointer to input matrix.
*/
__host__ int input_Loading(float N[][WIDTH2]) {

	FILE *fp;
	//fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/bigIntInput.txt", "r");
	fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/bigFloatInput.txt", "r");

	if (fp == NULL) {
		printf("Failed to open input file.\n");
		return(-1);
	}
	else {
		for (int i = 0; i < WIDTH1; i++)
			for (int j = 0; j < WIDTH2; j++)
				fscanf(fp, "%f", &N[i][j]);
	}

	fclose(fp);

	return 0;

}

/*!
* Host function of mask input loading that reads from a file.
* @param M pointer to mask matrix.
*/
__host__ int mask_Loading(float M[][MASK_WIDTH2]) {

	FILE *fp;
	//Sharpen_Mask has MW1=3, MW2=3 (int)
	//fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/Sharpen_Mask.txt", "r");
	//Sobel_Mask has MW1=9, MW2=9 (int)
	//fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/Sobel_Mask.txt", "r");
	//Box_blur_Mask has MW1=3, MW2=3 (float)
	//fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/Box_blur_Mask.txt", "r");
	//Gaussian_blur_Mask has MW1=7, MW2=7 (float)
	//fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/Gaussian_blur_Mask.txt", "r");
	//Stress_Mask1 has MW1=32, MW2=32 (float)
	//fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/Stress_Mask1.txt", "r");
	//Stress_Mask2 has MW1=77, MW2=77 (float)
	fp = fopen("/storage/pdarienzo/tirocinio/tesi/inputValues/Stress_Mask2.txt", "r");

	if (fp == NULL) {
		printf("Failed to open mask input file.\n");
		return(-1);
	}
	else {
		for (int i = 0; i < MASK_WIDTH1; i++)
			for (int j = 0; j < MASK_WIDTH2; j++)
				fscanf(fp, "%f", &M[i][j]);
	}

	fclose(fp);

	return 0;

}

/*!
* Host function of output printing on a file.
* @param P pointer to output matrix.
*/
__host__ int output_Writing(float P[][WIDTH2]) {

	FILE *fp;
	fp = fopen("/storage/pdarienzo/tirocinio/tesi/outputValues/parallelConvolution2DOutput.txt", "w");

	if (fp == NULL) {
		printf("Failed to open output file.\n");
		return(-1);
	}
	else {
		for (int i = 0; i < WIDTH1; i++) {
			for (int j = 0; j < WIDTH2; j++)
				fprintf(fp, "%6.3f\t", P[i][j]);
			fprintf(fp, "\n");
		}
	}

	fclose(fp);

	return 0;

}
