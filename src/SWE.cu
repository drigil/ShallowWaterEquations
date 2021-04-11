#include "SWE.cuh"

// To Do - 
// 1. Use cuda's math.h for speedup
// 2. Check if shared memory overlaps not causing issues

// KERNELS 
// WENO reconstruction - Refer Appendix A of paper


// WENO for flux for h component and the source term derivatives  (due to the bathymetry)
__device__ void WENOPosX(float3 v_iP, float3 v_i, float3 v_iN, float b_iP, float b_i, float b_iN, float3 &flux, float &source){
	// Get the smoothness indicator
	float3 beta0 = (v_iN - v_i) * (v_iN - v_i);
	float3 beta1 = (v_i - v_iP) * (v_i - v_iP);

	v_iP.y = v_iP.y - (0.25 * GRAVITY * b_iP * b_iP);
	v_i.y = v_i.y - (0.25 * GRAVITY * b_i * b_i);
	v_iN.y = v_iN.y - (0.25 * GRAVITY * b_iN * b_iN);

	// Get the stencil approximations
	float3 flux0 = 0.5 * v_i + 0.5 * v_iN;
	float3 flux1 = -0.5 * v_iP + 1.5 * v_i;

	float source_0 = 0.5 * b_i + 0.5 * b_iN;
	float source_1 = -0.5 * b_iP + 1.5 * b_i;

	// find the non-linear weights
	float3 a0 = (2.0f / 3.0f) / ((WENO_EPSILON + beta0) * (WENO_EPSILON + beta0));
	float3 a1 = (1.0f / 3.0f) / ((WENO_EPSILON + beta1) * (WENO_EPSILON + beta1));

	float3 aTot = a0 + a1;

	float3 weight0 = a0 / aTot;
	float3 weight1 = a1 / aTot;

	flux = weight0 * flux0 + weight1 * flux1;
	source = weight0.y * source_0 + weight1.y * source_1;
}

__device__ void WENONegX(float3 v_i, float3 v_iN, float3 v_iNN, float b_i, float b_iN, float b_iNN, float3 &flux, float &source){
	// Get the smoothness indicator
	float3 beta0 = (v_iN - v_i) * (v_iN - v_i);
	float3 beta1 = (v_iNN - v_iN) * (v_iNN - v_iN);

	v_i.y = v_i.y - (0.25 * GRAVITY * b_i * b_i);
	v_iN.y = v_iN.y - (0.25 * GRAVITY * b_iN * b_iN);
	v_iNN.y = v_iNN.y - (0.25 * GRAVITY * b_iNN * b_iNN);

	// Get the stencil approximations
	float3 flux0 = 0.5 * v_iN + 0.5 * v_i;
	float3 flux1 = -0.5 * v_iNN + 1.5 * v_iN;

	float source_0 = 0.5 * b_iN + 0.5 * b_i;
	float source_1 = -0.5 * b_iNN + 1.5 * b_iN;

	// find the non-linear weights
	float3 a0 = (2.0f / 3.0f) / ((WENO_EPSILON + beta0) * (WENO_EPSILON + beta0));
	float3 a1 = (1.0f / 3.0f) / ((WENO_EPSILON + beta1) * (WENO_EPSILON + beta1));

	float3 aTot = a0 + a1;

	float3 weight0 = a0 / aTot;
	float3 weight1 = a1 / aTot;

	flux = weight0 * flux0 + weight1 * flux1;
	source = weight0.y * source_0 + weight1.y * source_1;
}

__device__ void WENOPosY(float3 v_iP, float3 v_i, float3 v_iN, float b_iP, float b_i, float b_iN, float3 &flux, float &source){
	// Get the smoothness indicator
	float3 beta0 = (v_iN - v_i) * (v_iN - v_i);
	float3 beta1 = (v_i - v_iP) * (v_i - v_iP);

	v_iP.z = v_iP.z - (0.25 * GRAVITY * b_iP * b_iP);
	v_i.z = v_i.z - (0.25 * GRAVITY * b_i * b_i);
	v_iN.z = v_iN.z - (0.25 * GRAVITY * b_iN * b_iN);

	// Get the stencil approximations
	float3 flux0 = 0.5 * v_i + 0.5 * v_iN;
	float3 flux1 = -0.5 * v_iP + 1.5 * v_i;

	float source_0 = 0.5 * b_i + 0.5 * b_iN;
	float source_1 = -0.5 * b_iP + 1.5 * b_i;

	// find the non-linear weights
	float3 a0 = (2.0f / 3.0f) / ((WENO_EPSILON + beta0) * (WENO_EPSILON + beta0));
	float3 a1 = (1.0f / 3.0f) / ((WENO_EPSILON + beta1) * (WENO_EPSILON + beta1));

	float3 aTot = a0 + a1;

	float3 weight0 = a0 / aTot;
	float3 weight1 = a1 / aTot;

	flux = weight0 * flux0 + weight1 * flux1;
	source = weight0.z * source_0 + weight1.z * source_1;
}

__device__ void WENONegY(float3 v_i, float3 v_iN, float3 v_iNN, float b_i, float b_iN, float b_iNN, float3 &flux, float &source){
	// Get the smoothness indicator
	float3 beta0 = (v_iN - v_i) * (v_iN - v_i);
	float3 beta1 = (v_iNN - v_iN) * (v_iNN - v_iN);

	v_i.z = v_i.z - (0.25 * GRAVITY * b_i * b_i);
	v_iN.z = v_iN.z - (0.25 * GRAVITY * b_iN * b_iN);
	v_iNN.z = v_iNN.z - (0.25 * GRAVITY * b_iNN * b_iNN);

	// Get the stencil approximations
	float3 flux0 = 0.5 * v_iN + 0.5 * v_i;
	float3 flux1 = -0.5 * v_iNN + 1.5 * v_iN;

	float source_0 = 0.5 * b_iN + 0.5 * b_i;
	float source_1 = -0.5 * b_iNN + 1.5 * b_iN;

	// find the non-linear weights
	float3 a0 = (2.0f / 3.0f) / ((WENO_EPSILON + beta0) * (WENO_EPSILON + beta0));
	float3 a1 = (1.0f / 3.0f) / ((WENO_EPSILON + beta1) * (WENO_EPSILON + beta1));

	float3 aTot = a0 + a1;

	float3 weight0 = a0 / aTot;
	float3 weight1 = a1 / aTot;

	flux = weight0 * flux0 + weight1 * flux1;
	source = weight0.z * source_0 + weight1.z * source_1;
}


// Solve Shallow Water Equations
__global__ void applySWE(float* d_height, float* d_momentumU, float* d_momentumV, int* d_offsetX, int* d_offsetY){

	// Should total be 40.96KB for 32 X 32
	__shared__ float terrainArr[NUM_THREADS_Y][NUM_THREADS_X]; //Contains terrain point heights
	__shared__ float3 pointInfoArr[NUM_THREADS_Y][NUM_THREADS_X]; // Contains height and Eigen values 
	__shared__ float3 fluxFArr[NUM_THREADS_Y][NUM_THREADS_X]; // Contains F Flux values
	__shared__ float3 fluxGArr[NUM_THREADS_Y][NUM_THREADS_X]; // Contains G flux values

	// Getting global threadIDs
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	float height = d_height[x + y*gridDim.x*blockDim.x];
	float momentumU = (height != 0.0) ? d_momentumU[x + y*gridDim.x*blockDim.x] / height : 0.0; 
	float momentumV = (height != 0.0) ? d_momentumV[x + y*gridDim.x*blockDim.x] / height : 0.0;

	int offsetX = d_offsetX[x + y*gridDim.x*blockDim.x];
	int offsetY = d_offsetY[x + y*gridDim.x*blockDim.x];

	// Used to compute alpha
	float eigenX = abs(momentumU) + sqrt(GRAVITY * height);
	float eigenY = abs(momentumV) + sqrt(GRAVITY * height);

	float bathymetryVal = 1.0f; // Implement later using height map

	terrainArr[threadIdx.y][threadIdx.x] = bathymetryVal;
	pointInfoArr[threadIdx.y][threadIdx.x] = make_float3(height, eigenX, eigenY);

	__syncthreads(); // Fill block of threads with terrain heights, current point height and eigen values

	// Get corresponding local alphas from obtained eigen values
	float alphaX = eigenX;
	float alphaY = eigenY;

	for(int k = -1; k<=2; k++){
		alphaX = max(alphaX, pointInfoArr[threadIdx.y][threadIdx.x + k].y);
		alphaY = max(alphaY, pointInfoArr[threadIdx.y + k][threadIdx.x].z);	
	}

	// Find the 2nd part of the low-order Lax Friedrich flux for the h component
	float hFluxX = -0.5 * alphaX * (pointInfoArr[threadIdx.y][threadIdx.x + 1].x - height);
	float hFluxY = -0.5 * alphaY * (pointInfoArr[threadIdx.y + 1][threadIdx.x].x - height);

	// Find F and G fluxes
	float3 fluxF = make_float3(height * momentumU, height*momentumU*momentumU + GRAVITY * height * height / 2.0f, height * momentumU * momentumV);
	float3 fluxG = make_float3(height * momentumV, height*momentumU*momentumV, height*momentumV*momentumV + GRAVITY * height * height / 2.0f);

	// Get the Jacobian Matrix

	
	fluxFArr[threadIdx.y][threadIdx.x] = fluxF;
	fluxGArr[threadIdx.y][threadIdx.x] = fluxG;
	
	__syncthreads(); // Fill block of threads with flux valeus

	// Find the 1st part of the low-order Lax Friedrich flux for the h component
	hFluxX = hFluxX + 0.5 * (fluxF.x + fluxFArr[threadIdx.y][threadIdx.x + 1].x);
	hFluxY = hFluxY + 0.5 * (fluxG.x + fluxGArr[threadIdx.y + 1][threadIdx.x].x);

	// Find the second order flux derivative wrt x and y
	float3 dfdx = (1.0f / (2.0f * dx)) * (fluxFArr[threadIdx.y][threadIdx.x + 1] - fluxFArr[threadIdx.y][threadIdx.x - 1]);
	float3 dfdy = (1.0f / (2.0f * dy)) * (fluxGArr[threadIdx.y + 1][threadIdx.x] - fluxGArr[threadIdx.y - 1][threadIdx.x]);

	// Find second order source term

	// Find F tilde and G tilde from paper, the time averaged fluxes
	float3 fTilde = fluxF ;// + (dt/2.0f) * multiplication term - check from paper
	float3 gTilde = fluxG ;// + (dt/2.0f) * multiplication term - check from paper

	// Replace the x and y component of pointInfoArr with the flux values, hFluxX and hFluxY
	pointInfoArr[threadIdx.y][threadIdx.x].x = hFluxX;
	pointInfoArr[threadIdx.y][threadIdx.x].y = hFluxY;

	__syncthreads(); // Update the point info arr with flux components of th  height

	// Finding Gamma for checks later
	float gammaVal = -(height - (dt / dx) * (hFluxX - pointInfoArr[threadIdx.y][threadIdx.x - 1].x) - (dt / dy) * (hFluxY - pointInfoArr[threadIdx.y - 1][threadIdx.x].y));



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// WENO Reconstruction in X Direction
	fluxFArr[threadIdx.y][threadIdx.x] = fTilde;
	fluxGArr[threadIdx.y][threadIdx.x] = make_float3(height + bathymetryVal, momentumU, momentumV); // Reuse shared memory to store point set quantities again

	__syncthreads(); // Fill shared memory with point information and F Flux values for X direction reconstruction

	// Set flux boundaries
	if((threadIdx.x + offsetX >= 0) && (threadIdx.x + offsetX < NUM_THREADS_X)){
		fluxFArr[threadIdx.y][threadIdx.x + offsetX] = fTilde * make_float3(-1.0f, 1.0f, 1.0f); // Reverse flux at boundaries
	}

	__syncthreads();

	float3 outFluxPosX, outFluxNegX;
	float outSourcePosX, outSourceNegX;

	// Applying Lax Friedrich flux splitting
	// Positive flux part
	float3 fPosX_iP_j = 0.5 * (fluxFArr[threadIdx.y][threadIdx.x - 1] + alphaX * fluxGArr[threadIdx.y][threadIdx.x - 1]); // iP denotes i previous
	float3 fPosX_i_j =   0.5 * (fluxFArr[threadIdx.y][threadIdx.x]     + alphaX * fluxGArr[threadIdx.y][threadIdx.x]    );
	float3 fPosX_iN_j = 0.5 * (fluxFArr[threadIdx.y][threadIdx.x + 1] + alphaX * fluxGArr[threadIdx.y][threadIdx.x + 1]); //iN dentoes i next

	// Reconstruct positive X
	WENOPosX(fPosX_iP_j, fPosX_i_j, fPosX_iN_j, terrainArr[threadIdx.y][threadIdx.x-1], terrainArr[threadIdx.y][threadIdx.x], terrainArr[threadIdx.y][threadIdx.x+1], outFluxPosX, outSourcePosX);

	// Negative flux part
	float3 fNegX_i_j = 0.5 * (fluxFArr[threadIdx.y][threadIdx.x] - alphaX * fluxGArr[threadIdx.y][threadIdx.x]); // iP denotes i previous
	float3 fNegX_iN_j =   0.5 * (fluxFArr[threadIdx.y][threadIdx.x + 1]     + alphaX * fluxGArr[threadIdx.y][threadIdx.x + 1]    );
	float3 fNegX_iNN_j = 0.5 * (fluxFArr[threadIdx.y][threadIdx.x + 2] + alphaX * fluxGArr[threadIdx.y][threadIdx.x + 2]); //iN dentoes i next

	// Reconstruct negative X
	WENONegX(fNegX_i_j, fNegX_iN_j, fNegX_iNN_j, terrainArr[threadIdx.y][threadIdx.x], terrainArr[threadIdx.y][threadIdx.x + 1], terrainArr[threadIdx.y][threadIdx.x+2], outFluxNegX, outSourceNegX);

	float3 outFluxX = outFluxNegX + outFluxPosX;
	float outSourceX = outSourcePosX + outSourceNegX;



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// WENO Reconstruction in Y Direction

	// Store G tilde for WENO reconstruction along Y direction
	pointInfoArr[threadIdx.y][threadIdx.x] = gTilde;
	__syncthreads();	

	// Set Flux boundaries
	if (((threadIdx.y + offsetY) >= 0) && ((threadIdx.y + offsetY) < NUM_THREADS_Y)) {
		pointInfoArr[threadIdx.y + offsetY][threadIdx.x] = gTilde * make_float3(-1.0f, 1.0f, 1.0f);
	}

	__syncthreads();


	float3 outFluxPosY, outFluxNegY;
	float outSourcePosY, outSourceNegY;

	// Applying Lax Friedrich flux splitting
	// Positive flux part
	float3 fPosY_i_jP = 0.5 * (fluxFArr[threadIdx.y - 1][threadIdx.x] + alphaY * fluxGArr[threadIdx.y - 1][threadIdx.x]); // iP denotes i previous
	float3 fPosY_i_j =   0.5 * (fluxFArr[threadIdx.y][threadIdx.x]     + alphaY * fluxGArr[threadIdx.y][threadIdx.x]);
	float3 fPosY_i_jN = 0.5 * (fluxFArr[threadIdx.y + 1][threadIdx.x] + alphaY * fluxGArr[threadIdx.y + 1][threadIdx.x]); //iN dentoes i next

	// Reconstruct positive X
	WENOPosY(fPosY_iP_j, fPosY_i_j, fPosY_iN_j, terrainArr[threadIdx.y - 1][threadIdx.x], terrainArr[threadIdx.y][threadIdx.x], terrainArr[threadIdx.y + 1][threadIdx.x], outFluxPosY, outSourcePosY);

	// Negative flux part
	float3 fNegY_i_j = 0.5 * (fluxFArr[threadIdx.y][threadIdx.x] - alphaY * fluxGArr[threadIdx.y][threadIdx.x]); // iP denotes i previous
	float3 fNegY_i_jN =   0.5 * (fluxFArr[threadIdx.y + 1][threadIdx.x] + alphaY * fluxGArr[threadIdx.y + 1][threadIdx.x]);
	float3 fNegY_i_jNN = 0.5 * (fluxFArr[threadIdx.y + 2][threadIdx.x] + alphaY * fluxGArr[threadIdx.y + 2][threadIdx.x]); //iN dentoes i next

	// Reconstruct negative X
	WENONegY(fNegY_i_j, fNegY_i_jN, fNegY_i_jNN, terrainArr[threadIdx.y][threadIdx.x], terrainArr[threadIdx.y + 1][threadIdx.x], terrainArr[threadIdx.y + 2][threadIdx.x], outFluxNegY, outSourceNegY);

	float3 outFluxY = outFluxNegY + outFluxPosY;
	float outSourceY = outSourcePosY + outSourceNegY;

}




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// Constructor
SWE::SWE(int numPointsX, int numPointsY){

	this->numPointsX = numPointsX;
	this->numPointsY = numPointsY;
	// Allocate memory to host variables
	// Allocating height and momentum variables (+2 for boundary?)
	// U variable in Equation
	h_height = (float*)malloc(sizeof(float) * (numPointsX + 2) * (numPointsY + 2)); 
	h_momentumU = (float*)malloc(sizeof(float) * (numPointsX + 2) * (numPointsY + 2)); 
	h_momentumV = (float*)malloc(sizeof(float) * (numPointsX + 2) * (numPointsY + 2)); 

	// Allocating flux terms (why +1?)
	// F and G variables in formula
	h_Fh = (float*)malloc(sizeof(numPointsX + 1) * (numPointsY + 1)); 
	h_Fhu = (float*)malloc(sizeof(numPointsX + 1) * (numPointsY + 1));
	h_Fhv = (float*)malloc(sizeof(numPointsX + 1) * (numPointsY + 1));
	h_Gh = (float*)malloc(sizeof(numPointsX + 1) * (numPointsY + 1));
	h_Ghu = (float*)malloc(sizeof(numPointsX + 1) * (numPointsY + 1));
	h_Ghv = (float*)malloc(sizeof(numPointsX + 1) * (numPointsY + 1));

	// Allocating max height, max velocity and charachteristic velocity
	h_maxHeight = new float;
	h_maxVelocity = new float;
	h_characteristicVelocity = new float;

	// Allocating memory for offsets
	h_offsetX = (int*)malloc(sizeof(numPointsX) * (numPointsY)); 
	h_offsetY = (int*)malloc(sizeof(numPointsX) * (numPointsY)); 

	// Allocate memory to device variables
	checkCudaErrors(cudaMalloc((void**)&d_height, (numPointsX+2)*(numPointsY+2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_momentumU, (numPointsX+2)*(numPointsY+2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_momentumV, (numPointsX+2)*(numPointsY+2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Fh, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Fhu, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Fhv, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Gh, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Ghu, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Ghv, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_maxHeight, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_maxVelocity, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_characteristicVelocity, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_offsetX, (numPointsX)*(numPointsY)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_offsetY, (numPointsX)*(numPointsY)*sizeof(int)));


	// Initializing device variables
	checkCudaErrors(cudaMemset(d_height, 0, (numPointsX+2)*(numPointsY+2)*sizeof(float)));
	checkCudaErrors(cudaMemset(d_momentumU, 0, (numPointsX+2)*(numPointsY+2)*sizeof(float)));
	checkCudaErrors(cudaMemset(d_momentumV, 0, (numPointsX+2)*(numPointsY+2)*sizeof(float)));
	checkCudaErrors(cudaMemset(d_Fh, 0, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMemset(d_Gh, 0, (numPointsX+1)*(numPointsY+1)*sizeof(float)));
	checkCudaErrors(cudaMemset(d_maxHeight, 0, sizeof(float)));
	checkCudaErrors(cudaMemset(d_maxVelocity, 0, sizeof(float)));
	checkCudaErrors(cudaMemset(d_characteristicVelocity, 0.001f, sizeof(float)));
	checkCudaErrors(cudaMemset(d_offsetX, 0, (numPointsX)*(numPointsY)*sizeof(int)));
	checkCudaErrors(cudaMemset(d_offsetY, 0, (numPointsX)*(numPointsY)*sizeof(int)));

}

// Set the starting conditions
void SWE::setInitialConditions(int conditionNum){
	switch(conditionNum){
		case 0:
			for(int i = 0; i < numPointsX; i++){
				for(int j = 0 ; j < numPointsY; j++){
					if(i > numPointsX/4 && i < 3*numPointsX/4 && j > numPointsY/4 && j < 3*numPointsY/4){
						h_height[i + j * (numPointsX + 2)] = 5.5f;
					}
					else{
						h_height[i + j * (numPointsX + 2)] = 1.0f;
				}
			}
		}
	}
}


