#include <iostream>
#include "SWE.cuh"

int main(){
	int numPointsX = 63;
	int numPointsY = 39;
	int conditionNum = 0;

	SWE swe(numPointsX, numPointsY);
	swe.setInitialConditions(conditionNum);

	for(int i = 0; i<numPointsX; i++){
		for(int j = 0; j<numPointsY; j++){
			printf("%f ", swe.h_height[i + j * (numPointsX + 2)]);
		}
		printf(" \n");
	}
	
	swe.simulate();

	cudaError_t kernelErr = cudaGetLastError();
	if(kernelErr!=cudaSuccess){
		printf("Error: %s\n", cudaGetErrorString(kernelErr));
	}


	printf("After running simulation \n");
	for(int i = 0; i<numPointsX; i++){
		for(int j = 0; j<numPointsY; j++){
			printf("%f ", swe.h_height[i + j * (numPointsX + 2)]);
		}
		printf(" \n");
	}
	
}