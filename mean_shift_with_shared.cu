#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

#define N 600                       //total points
#define D 2                         //dimensions
#define STANDARD_DEVIATION 1        //standard deviation
#define EPSILON 1e-4                //converge critirion

#define POINTS_FILE "x.bin"         //name of the file with the points




__global__ void meanShift(double *x0, double *y) {
    int index_y = blockIdx.x*D;       //y index
    int index_x = threadIdx.x*D;      //x index

    int i;
    double y_new[D], y_prev[D], m[D], m_norm;
    __shared__ double x[N*D];           //use shared memory for the x matrix
    __shared__ double denominator[N];   //variable to store the denominator sum
    __shared__ double numerator[N*D];   //variable to store the numerator sum



    for(int i=0;i<D;i++) {
        //init x matrix
        x[index_x + i] = x0[index_x + i];
    }


    __syncthreads();


    for(int i=0;i<D;i++) {
        //init y_prev
        y_prev[i] = x[index_y + i];
    }


    do{
        //do the subtraction
        for(i=0;i<D;i++) {
            y_new[i] = y_prev[i] - x[index_x + i];
        }


        //calculate norm
        denominator[threadIdx.x] = norm(D, y_new);

        if(denominator[threadIdx.x] <= STANDARD_DEVIATION) {
            //take the square
            denominator[threadIdx.x] *= denominator[threadIdx.x];

            //calculate the Gaussian kernel
            denominator[threadIdx.x] = exp(-denominator[threadIdx.x] / (2*STANDARD_DEVIATION));

            //calculate the numerator (for every j)
            for(i=0;i<D;i++) {
                numerator[index_x + i] = denominator[threadIdx.x] * x[index_x + i];
            }
        }
        else {
            denominator[threadIdx.x] = 0;
            for(i=0;i<D;i++) {
                numerator[index_x + i] = 0;
            }
        }


        __syncthreads();
        //reduction || do the sums
        for (unsigned int s=1;s<N;s*=2) {
            int index = 2 * s * threadIdx.x;
            if (index < N && (index + s) < N) {
                //denominator sum
                denominator[index] += denominator[index + s];

                //numerator sum
                for(i=0;i<D;i++) {
                    numerator[index*D + i] += numerator[(index + s)*D + i];
                }
            }
            __syncthreads();
        }


        for(int i=0;i<D;i++) {
            //calculate new y (y^k+1)
            y_new[i] = numerator[i] / denominator[0];

            //calculate m
            m[i] = y_new[i] - y_prev[i];

            //init y_prev for the next iteration
            y_prev[i] = y_new[i];
        }

        //calculate norm(m)
        m_norm = norm(D, m);

    } while(m_norm >= EPSILON);


    //return the final y
    for(i=0;i<D;i++) {
        y[index_y + i] = y_new[i];
    }

}



////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Main function /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    double *x, *y, *dev_x, *dev_y;
    int read;
    FILE *points, *result;
    cudaError_t error;



    //allocate memory for the x matrix
    x = (double*) malloc(N*D*sizeof(double));
    if(x == NULL) {
        fprintf(stderr, "malloc fail.\n");
        exit(1);
    }


    //allocate memory for the y matrix || result
    y = (double*) malloc(N*D*sizeof(double));
    if(y == NULL) {
        fprintf(stderr, "malloc fail.\n");
        exit(1);
    }


    //open the points file
    points = fopen(POINTS_FILE,"rb");
    if(points == NULL) {
      fprintf(stderr, "Unable to open file\n");
      exit(1);
    }


    //read the points data
    read = fread(x,sizeof(double),N*D,points);
    if (read != N*D) {
        fprintf(stderr, "Unable to read data\n");
        exit(1);
    }


    //allocate device memory for the x matrix
    if(cudaSuccess != cudaMalloc((void**)&dev_x, N*D*sizeof(double))) {
        fprintf(stderr, "cudaMalloc fail.\n");
        exit(1);
    }


    //allocate device memory for the y matrix
    if(cudaSuccess != cudaMalloc((void**)&dev_y, N*D*sizeof(double))) {
        fprintf(stderr, "cudaMalloc fail.\n");
        exit(1);
    }


    //move x matrix to the device
    error = cudaMemcpy(dev_x, x, N*D*sizeof(double), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        fprintf(stderr, "Moving data to device fail.\n");
        exit(1);
    }


    //variables for time measurement
    struct timeval tval_before, tval_after, tval_result;


    //blocksize and thread size
    dim3 dimGrid(N);
    dim3 dimBlock(N);


    //start time measurement
    gettimeofday(&tval_before, NULL);


    //calculate the mean shift
    meanShift<<<dimGrid, dimBlock>>>(dev_x, dev_y);


    //wait for meanShift to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
        fprintf(stderr, "Fail synchronize\n");
        exit(1);
    }


    //end time measurement
    gettimeofday (&tval_after, NULL);


    //calculate the execute time
    timersub(&tval_after, &tval_before, &tval_result);


    //time in seconds
    //printf("Took %ld.%06ld sec\n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    //time in micro seconds
    printf("Took %ld micro seconds\n", (long int) ((tval_result.tv_sec) * 1.0e6) + tval_result.tv_usec);



    //take the y matrix values from the device
    error = cudaMemcpy(y, dev_y, N*D*sizeof(double), cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        fprintf(stderr, "Moving data to device fail.\n");
        exit(1);
    }


    result = fopen("result.txt", "a");


    for(int i=0;i<N;i++) {
        for(int j=0;j<D;j++) {
            fprintf(result, "%lf ", y[i*D + j]);
        }
        fprintf(result, "\n");
    }


    free(x);
    free(y);
    cudaFree(dev_x);
    cudaFree(dev_y);
    fclose(points);
    fclose(result);
}
