#include "SWE.cuh"

// To Do - 
// 1. Use cuda's math.h for speedup
// 2. Check if shared memory overlaps not causing issues - Done
// 3. Check bcCount - Done
// 4. Matmul and some matrices left - Done
// 5. Deal with shared memory out of bounds access - Done
// 6. Global index - Done
// 7. Offsets - Done

// Errors Occurred
// 1. Too many resources - Solved by reducing block dimensions
// 2. Warp out of bound - Solved (Hopefully)
// 3. Global Indexing incorrect - Solved (Hopefully)
// 4. Offsets not set - Solved (hopefully)

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

	float3 weight0 = a0 / (a0 + a1);
	float3 weight1 = a1 / (a0 + a1);

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

	float3 weight0 = a0 / (a0 + a1);
	float3 weight1 = a1 / (a0 + a1);

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

	float3 weight0 = a0 / (a0 + a1);
	float3 weight1 = a1 / (a0 + a1);

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

	float3 weight0 = a0 / (a0 + a1);
	float3 weight1 = a1 / (a0 + a1);

	flux = weight0 * flux0 + weight1 * flux1;
	source = weight0.z * source_0 + weight1.z * source_1;
}


// Solve Shallow Water Equations
__global__ void applySWE(int numPointsX, int numPointsY, float* d_height, float* d_momentumU, float* d_momentumV, int* d_offsetX, int* d_offsetY, float* d_height_out, float* d_momentumU_out, float* d_momentumV_out){

	// +2 for ahead and -2 for back values
	__shared__ float terrainArr[NUM_THREADS_Y + BOUNDARY_CELL_COUNT][NUM_THREADS_X + BOUNDARY_CELL_COUNT]; //Contains terrain point heights
	__shared__ float3 pointInfoArr[NUM_THREADS_Y + BOUNDARY_CELL_COUNT][NUM_THREADS_X + BOUNDARY_CELL_COUNT]; // Contains height and Eigen values 
	__shared__ float3 fluxFArr[NUM_THREADS_Y + BOUNDARY_CELL_COUNT][NUM_THREADS_X + BOUNDARY_CELL_COUNT]; // Contains F Flux values
	__shared__ float3 fluxGArr[NUM_THREADS_Y + BOUNDARY_CELL_COUNT][NUM_THREADS_X + BOUNDARY_CELL_COUNT]; // Contains G flux values

	// Getting global threadIDs
	// int x = threadIdx.x + blockIdx.x*blockDim.x;
	// int y = threadIdx.y + blockIdx.y*blockDim.y;

	// each group invocation deals with (THREADS_X - 2 * bcCount) * (THREADS_Y - 2 * bcCount) amount of internal data
	// Look at section 3.3 to see how threading works

	int globalX = blockIdx.x * (NUM_THREADS_X - 2 * BOUNDARY_CELL_COUNT) + threadIdx.x;
	int globalY = blockIdx.y * (NUM_THREADS_Y - 2 * BOUNDARY_CELL_COUNT) + threadIdx.y;

	int localX = threadIdx.x + 2; //Shift for boundary data computations
	int localY = threadIdx.y + 2;

	float height = d_height[globalX + globalY * numPointsX];
	float momentumU = (height != 0.0) ? d_momentumU[globalX + globalY * numPointsX] / height : 0.0; 
	float momentumV = (height != 0.0) ? d_momentumV[globalX + globalY * numPointsX] / height : 0.0;

	float3 uCurr = make_float3(height, momentumU * height, momentumV * height);

	int offsetX = d_offsetX[globalX + globalY * numPointsX];
	int offsetY = d_offsetY[globalX + globalY * numPointsX];

	// Used to compute alpha
	float eigenX = abs(momentumU) + sqrt(GRAVITY * height);
	float eigenY = abs(momentumV) + sqrt(GRAVITY * height);

	float bathymetryVal = 0.0f; // Implement later using height map

	terrainArr[localY][localX] = bathymetryVal;
	pointInfoArr[localY][localX] = make_float3(height, eigenX, eigenY);

	// Initialize boundary values
	if(threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == NUM_THREADS_X - 1 || threadIdx.x == NUM_THREADS_X - 2 || threadIdx.y == 0 || threadIdx.y == 1 || threadIdx.y == NUM_THREADS_Y - 1 || threadIdx.y == NUM_THREADS_Y - 2){
		terrainArr[threadIdx.y][threadIdx.x] = 0.0f;
		pointInfoArr[threadIdx.y][threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
		fluxFArr[threadIdx.y][threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
		fluxGArr[threadIdx.y][threadIdx.x] = make_float3(0.0f, 0.0f, 0.0f);
			
	}

	__syncthreads(); // Fill block of threads with terrain heights, current point height and eigen values

	// Get corresponding local alphas from obtained eigen values
	float alphaX = eigenX;
	float alphaY = eigenY;

	for(int k = -1; k<=2; k++){
		alphaX = max(alphaX, pointInfoArr[localY][localX + k].y);
		alphaY = max(alphaY, pointInfoArr[localY + k][localX].z);	
	}

	// Find the 2nd part of the low-order Lax Friedrich flux for the h component
	float hFluxX = -0.5 * alphaX * (pointInfoArr[localY][localX + 1].x - height);
	float hFluxY = -0.5 * alphaY * (pointInfoArr[localY + 1][localX].x - height);

	// Find F and G fluxes
	float3 fluxF = make_float3(height * momentumU, height*momentumU*momentumU + GRAVITY * height * height / 2.0f, height * momentumU * momentumV);
	float3 fluxG = make_float3(height * momentumV, height*momentumU*momentumV, height*momentumV*momentumV + GRAVITY * height * height / 2.0f);

	// Get the Jacobian Matrix
	float3 dfduR1 = make_float3(0.0f, 1.0f, 0.0f);
	float3 dfduR2 = make_float3(-momentumU * momentumU + GRAVITY * height, 2.0f * momentumU, 0.0f);
	float3 dfduR3 = make_float3(-momentumU * momentumV, momentumV, momentumU);

	float3 dgduR1 = make_float3(0.0f, 0.0f, 1.0f);
	float3 dgduR2 = make_float3( -momentumU * momentumV, momentumV, momentumU);
	float3 dgduR3 = make_float3(-momentumV * momentumV + GRAVITY * height, 0.0f, 2.0f * momentumV);
	
	fluxFArr[localY][localX] = fluxF;
	fluxGArr[localY][localX] = fluxG;
	
	__syncthreads(); // Fill block of threads with flux valeus

	// Find the 1st part of the low-order Lax Friedrich flux for the h component
	hFluxX = hFluxX + 0.5 * (fluxF.x + fluxFArr[localY][localX + 1].x);
	hFluxY = hFluxY + 0.5 * (fluxG.x + fluxGArr[localY + 1][localX].x);

	// Find the second order flux derivative wrt x and y
	float3 dfdx = (1.0f / (2.0f * dx)) * (fluxFArr[localY][localX + 1] - fluxFArr[localY][localX - 1]);
	float3 dgdy = (1.0f / (2.0f * dy)) * (fluxGArr[localY + 1][localX] - fluxGArr[localY - 1][localX]);

	// Find second order source term
	float3 sourceLow = make_float3(0.0f, (1.0f / dx) * ((0.25 * GRAVITY * pow(terrainArr[localY][localX + 1], 2)) - (0.25 * GRAVITY * pow(terrainArr[localY][localX - 1], 2))) - GRAVITY * (height + bathymetryVal) * (1.0 / (2.0 * dx)) * (terrainArr[localY][localX + 1] - terrainArr[localY][localX - 1]), (1.0 / dy) * ((0.25 * GRAVITY * pow(terrainArr[localY + 1][localX], 2)) - (0.25 * GRAVITY * pow(terrainArr[localY - 1][localX], 2))) - GRAVITY * (height + bathymetryVal) * (1.0 / (2.0 * dy)) * (terrainArr[localY + 1][localX] - terrainArr[localY - 1][localX]));

	// Find F tilde and G tilde from paper, the time averaged fluxes
	//Applying matrix multiplication
	float3 fTilde = fluxF + ((dt / 2.0f) * make_float3(sum(dfduR1 * (sourceLow - dfdx - dgdy)), sum(dfduR2 * (sourceLow - dfdx - dgdy)), sum(dfduR3 * (sourceLow - dfdx - dgdy))));
	float3 gTilde = fluxG + ((dt / 2.0f) * make_float3(sum(dgduR1 * (sourceLow - dfdx - dgdy)), sum(dgduR2 * (sourceLow - dfdx - dgdy)), sum(dgduR3 * (sourceLow - dfdx - dgdy))));

	// Replace the x and y component of pointInfoArr with the flux values, hFluxX and hFluxY
	pointInfoArr[localY][localX].x = hFluxX;
	pointInfoArr[localY][localX].y = hFluxY;

	__syncthreads(); // Update the point info arr with flux components of th  height

	// Finding Gamma for checks later
	float gammaVal = -(height - (dt / dx) * (hFluxX - pointInfoArr[localY][localX - 1].x) - (dt / dy) * (hFluxY - pointInfoArr[localY - 1][localX].y));

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// WENO Reconstruction in X Direction
	fluxFArr[localY][localX] = fTilde;
	fluxGArr[localY][localX] = make_float3(height + bathymetryVal, uCurr.y, uCurr.z); // Reuse shared memory to store point set quantities again

	__syncthreads(); // Fill shared memory with point information and F Flux values for X direction reconstruction

	// Set flux boundaries
	if((offsetX!=0) && (threadIdx.x >= BOUNDARY_CELL_COUNT) && (threadIdx.x < NUM_THREADS_X - BOUNDARY_CELL_COUNT)){ // Check
		fluxFArr[localY][localX + offsetX] = fTilde * make_float3(-1.0f, 1.0f, 1.0f); // Reverse flux at boundaries
	}

	__syncthreads();

	float3 outFluxPosX, outFluxNegX;
	float outSourcePosX, outSourceNegX;

	// Applying Lax Friedrich flux splitting
	// Positive flux part
	float3 fPosX_iP_j = 0.5 * (fluxFArr[localY][localX - 1] + alphaX * fluxGArr[localY][localX - 1]); // iP denotes i previous
	float3 fPosX_i_j =   0.5 * (fluxFArr[localY][localX]     + alphaX * fluxGArr[localY][localX]    );
	float3 fPosX_iN_j = 0.5 * (fluxFArr[localY][localX + 1] + alphaX * fluxGArr[localY][localX + 1]); //iN dentoes i next

	// Reconstruct positive X
	WENOPosX(fPosX_iP_j, fPosX_i_j, fPosX_iN_j, terrainArr[localY][localX-1], terrainArr[localY][localX], terrainArr[localY][localX + 1], outFluxPosX, outSourcePosX);

	// Negative flux part
	float3 fNegX_i_j = 0.5 * (fluxFArr[localY][localX] - alphaX * fluxGArr[localY][localX]); // iP denotes i previous
	float3 fNegX_iN_j =   0.5 * (fluxFArr[localY][localX + 1] - alphaX * fluxGArr[localY][localX + 1]);
	float3 fNegX_iNN_j = 0.5 * (fluxFArr[localY][localX + 2] - alphaX * fluxGArr[localY][localX + 2]); //iN dentoes i next

	// Reconstruct negative X
	WENONegX(fNegX_i_j, fNegX_iN_j, fNegX_iNN_j, terrainArr[localY][localX], terrainArr[localY][localX + 1], terrainArr[localY][localX + 2], outFluxNegX, outSourceNegX);

	float3 outFluxX = outFluxNegX + outFluxPosX;
	float outSourceX = 0.5 * outSourcePosX + 0.5 * outSourceNegX;



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// WENO Reconstruction in Y Direction
	// Store G tilde for WENO reconstruction along Y direction
	pointInfoArr[localY][localX] = gTilde;
	__syncthreads();	

	// Set Flux boundaries
	if ((offsetY!=0) && (threadIdx.y >= BOUNDARY_CELL_COUNT) && (threadIdx.y < NUM_THREADS_Y - BOUNDARY_CELL_COUNT)) {
		pointInfoArr[localY + offsetY][localX] = gTilde * make_float3(-1.0f, 1.0f, 1.0f);
	}

	__syncthreads();


	float3 outFluxPosY, outFluxNegY;
	float outSourcePosY, outSourceNegY;

	// Applying Lax Friedrich flux splitting
	// Positive flux part
	float3 fPosY_i_jP = 0.5 * (pointInfoArr[localY - 1][localX] + alphaY * fluxGArr[localY - 1][localX]); // iP denotes i previous
	float3 fPosY_i_j =   0.5 * (pointInfoArr[localY][localX] + alphaY * fluxGArr[localY][localX]);
	float3 fPosY_i_jN = 0.5 * (pointInfoArr[localY + 1][localX] + alphaY * fluxGArr[localY + 1][localX]); //iN dentoes i next

	// Reconstruct positive X
	WENOPosY(fPosY_i_jP, fPosY_i_j, fPosY_i_jN, terrainArr[localY - 1][localX], terrainArr[localY][localX], terrainArr[localY + 1][localX], outFluxPosY, outSourcePosY);

	// Negative flux part
	float3 fNegY_i_j = 0.5 * (pointInfoArr[localY][localX] - alphaY * fluxGArr[localY][localX]); // iP denotes i previous
	float3 fNegY_i_jN =   0.5 * (pointInfoArr[localY + 1][localX] - alphaY * fluxGArr[localY + 1][localX]);
	float3 fNegY_i_jNN = 0.5 * (pointInfoArr[localY + 2][localX] - alphaY * fluxGArr[localY + 2][localX]); //iN dentoes i next

	// Reconstruct negative X
	WENONegY(fNegY_i_j, fNegY_i_jN, fNegY_i_jNN, terrainArr[localY][localX], terrainArr[localY + 1][localX], terrainArr[localY + 2][localX], outFluxNegY, outSourceNegY);

	float3 outFluxY = outFluxNegY + outFluxPosY;
	float outSourceY = 0.5 * outSourcePosY + 0.5 * outSourceNegY;


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Positivity Preservation - For wetting and Drying
	float f_i = (dt/dx) * (outFluxX.x - hFluxX);
	float f_j = (dt/dy) * (outFluxY.x - hFluxY);

	// Store results in shared memory
	fluxFArr[localY][localX].x = f_i;
	fluxFArr[localY][localX].y = f_j;

	__syncthreads();

	// Values to be checked
	float f_iN_j = -f_i;
	float f_iP_j = fluxFArr[localY][localX - 1].x;
	float f_i_jN = -f_j;
	float f_i_jP = fluxFArr[localX - 1][localX].y;

	// To avoid branching due to conditional statements
	int alpha = f_iN_j < 0;
	int beta = f_iP_j < 0;
	int gamma = f_i_jN < 0;
	int delta = f_i_jP < 0;

	float R = (alpha * f_iN_j + beta * f_iP_j + gamma * f_i_jN + delta * f_i_jP);
	float Q = min(1.0f, (R != 0.0f) ? gammaVal / R : 0.0f);

	float hat_r = (1 - alpha) + alpha * Q;
	float hat_l = (1 - beta) + beta * Q;
	float hat_u = (1 - gamma) + gamma * Q;
	float hat_d = (1 - delta) + delta * Q;

	pointInfoArr[localY][localX].x = hat_l;
	pointInfoArr[localY][localX].y = hat_d;

	__syncthreads();

	float theta_i = min(hat_r, pointInfoArr[localY][localX + 1].x);
	float theta_j = min(hat_u, pointInfoArr[localY + 1][localX].y);
	
	// Final h flux
	float fTildeFinal = theta_i * (outFluxX.x - hFluxX) + hFluxX;
	float gTildeFinal = theta_j * (outFluxY.x - hFluxY) + hFluxY;




	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Time Integration

	// Store final fluxes in x and y directions
	float3 finalFluxX = make_float3(fTildeFinal, outFluxX.y, outFluxX.z);
	float3 finalFluxY = make_float3(gTildeFinal, outFluxY.y, outFluxY.z);
	fluxFArr[localY][localX] = finalFluxX;
	fluxGArr[localY][localX] = finalFluxY;

	// Store source terms in shared memory
	terrainArr[localY][localX] = outSourceX;
	pointInfoArr[localY][localX].z = outSourceY;

	__syncthreads();

	float3 uNew = uCurr - ((dt / dx) * (finalFluxX - fluxFArr[localY][localX - 1])) - (dt/dy) * (finalFluxY -fluxGArr[localY - 1][localX]) - (GRAVITY * (height + bathymetryVal) * make_float3(0.0, (dt/dx) * (outSourceX - terrainArr[localY][localX - 1]), (dt/dy) * (outSourceY - pointInfoArr[localY - 1][localX].z)));
	// Find the veloctities for the current cell
	float heightTerm = uNew.x * uNew.x * uNew.x * uNew.x;

	// Desinguralize velocities
	float momentumUTerm = (uNew.x < DESING_EPSILON) ? sqrt(2.0f) * uNew.x * uNew.y / sqrt((heightTerm + max(heightTerm, DESING_EPSILON))):uNew.y / uNew.x;
	float momentumVTerm = (uNew.x < DESING_EPSILON) ? sqrt(2.0f) * uNew.x * uNew.z / sqrt((heightTerm + max(heightTerm, DESING_EPSILON))):uNew.z / uNew.x;

	// Consistency requirement
	uNew.x = max(0.0, uNew.x);
	uNew.y = uNew.x * momentumUTerm;
	uNew.z = uNew.x * momentumVTerm;

	// printf("height, momentumU and momentumV are %f %f %f\n", uNew.x, uNew.y, uNew.z);

	// Check if inner domain cell
	bool isDomainInner = (globalX >= BOUNDARY_CELL_COUNT) && (globalX<(numPointsX + BOUNDARY_CELL_COUNT)) && (globalY >= BOUNDARY_CELL_COUNT) && (globalY < (numPointsY + BOUNDARY_CELL_COUNT));

	// Check if in inner domain of current patch
	bool isPatchInner = (threadIdx.x >= BOUNDARY_CELL_COUNT) && (threadIdx.x < (NUM_THREADS_X - BOUNDARY_CELL_COUNT)) && (threadIdx.y >= BOUNDARY_CELL_COUNT) && (threadIdx.y < (NUM_THREADS_Y - BOUNDARY_CELL_COUNT));
	
	// d_height_out[globalX + globalY*numPointsX] = uNew.x;
	// d_momentumU_out[globalX + globalY*numPointsX] = uNew.y;
	// d_momentumV_out[globalX + globalY*numPointsX] = uNew.z;

	if(isDomainInner && isPatchInner){
		d_height_out[globalX + globalY*numPointsX] = uNew.x;
		d_momentumU_out[globalX + globalY*numPointsX] = uNew.y;
		d_momentumV_out[globalX + globalY*numPointsX] = uNew.z;
		// Assign offsets too

		// Set the boundary conditions
		if(offsetX!=0 && threadIdx.x >= BOUNDARY_CELL_COUNT && threadIdx.x < NUM_THREADS_X - BOUNDARY_CELL_COUNT){
			d_height_out[globalX + offsetX + globalY*numPointsX] = uNew.x;
			d_momentumU_out[globalX + offsetX + globalY*numPointsX] = -uNew.y;
			d_momentumV_out[globalX + offsetX + globalY*numPointsX] = -uNew.z;
		}
		
		if(offsetY!=0 && threadIdx.y >= BOUNDARY_CELL_COUNT && threadIdx.y < NUM_THREADS_Y - BOUNDARY_CELL_COUNT){
			d_height_out[globalX + (globalY + offsetY)*numPointsX] = uNew.x;
			d_momentumU_out[globalX + (globalY + offsetY)*numPointsX] = -uNew.y;
			d_momentumV_out[globalX + (globalY + offsetY)*numPointsX] = -uNew.z;
		}

	}
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
	h_height = (float*)malloc(sizeof(float) * (numPointsX) * (numPointsY)); 
	h_momentumU = (float*)malloc(sizeof(float) * (numPointsX) * (numPointsY)); 
	h_momentumV = (float*)malloc(sizeof(float) * (numPointsX) * (numPointsY)); 

	h_height_out = (float*)malloc(sizeof(float) * (numPointsX) * (numPointsY)); 
	h_momentumU_out = (float*)malloc(sizeof(float) * (numPointsX) * (numPointsY)); 
	h_momentumV_out = (float*)malloc(sizeof(float) * (numPointsX) * (numPointsY)); 

	// Allocating memory for offsets
	h_offsetX = (int*)malloc(sizeof(int) * (numPointsX) * (numPointsY)); 
	h_offsetY = (int*)malloc(sizeof(int) * (numPointsX) * (numPointsY)); 

	int3 blockId = make_int3(((numPointsX - 1- (NUM_THREADS_X - 1))/ (NUM_THREADS_X - 2*BOUNDARY_CELL_COUNT)) + 1, ((numPointsY - 1 - (NUM_THREADS_Y - 1))/ (NUM_THREADS_Y - 2*BOUNDARY_CELL_COUNT)) + 1, 1);
	int3 threadId = make_int3(NUM_THREADS_X, NUM_THREADS_Y, 1);
	int globalX;
	int globalY;

	// Set offset values
	for (int bIdx = 0; bIdx <blockId.x; bIdx ++){
		for(int bIdy = 0; bIdy < blockId.y; bIdy++){
			for(int tIdx = 0; tIdx < threadId.x; tIdx++){
				for(int tIdy = 0; tIdy < threadId.y; tIdy++){
					globalX = bIdx * (NUM_THREADS_X - 2 * BOUNDARY_CELL_COUNT) + tIdx;
					globalY = bIdy * (NUM_THREADS_Y - 2 * BOUNDARY_CELL_COUNT) + tIdy;

					if((tIdx>=BOUNDARY_CELL_COUNT) && (tIdx<2*BOUNDARY_CELL_COUNT)){
						h_offsetX[globalX + globalY*numPointsX] = 2*(BOUNDARY_CELL_COUNT - (tIdx + 1)) + 1;
					}

					if((tIdx >= NUM_THREADS_X - 2*BOUNDARY_CELL_COUNT) && (tIdx< NUM_THREADS_X - BOUNDARY_CELL_COUNT)){
						h_offsetX[globalX + globalY*numPointsX] = 2 * (NUM_THREADS_X - (tIdx) - 4) - 1;
					}

					if((tIdy>=BOUNDARY_CELL_COUNT) && (tIdy<2*BOUNDARY_CELL_COUNT)){
						h_offsetY[globalX + globalY*numPointsX] = 2*(BOUNDARY_CELL_COUNT - (tIdy + 1)) + 1;
					}

					if((tIdy >= NUM_THREADS_Y - 2*BOUNDARY_CELL_COUNT) && (tIdy< NUM_THREADS_Y - BOUNDARY_CELL_COUNT)){
						h_offsetY[globalX + globalY*numPointsX] = 2 * (NUM_THREADS_Y - (tIdy) - 4) - 1;
					}

				}
			}
		}
	}

	// Allocate memory to device variables
	checkCudaErrors(cudaMalloc((void**)&d_height, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_momentumU, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_momentumV, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_offsetX, (numPointsX)*(numPointsY)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_offsetY, (numPointsX)*(numPointsY)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_height_out, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_momentumU_out, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_momentumV_out, (numPointsX)*(numPointsY)*sizeof(float)));

	// Initializing device variables
	checkCudaErrors(cudaMemset(d_height, 0.0f, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMemset(d_momentumU, 0.0f, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMemset(d_momentumV, 0.0f, (numPointsX)*(numPointsY)*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_offsetX, h_offsetX, (numPointsX)*(numPointsY)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_offsetY, h_offsetY, (numPointsX)*(numPointsY)*sizeof(int), cudaMemcpyHostToDevice));

}

// Set the starting conditions
void SWE::setInitialConditions(int conditionNum){
	switch(conditionNum){
		case 0:
			for(int i = 0; i < numPointsX; i++){
				for(int j = 0 ; j < numPointsY; j++){
					if(i > numPointsX/4 && i < 3*numPointsX/4 && j > numPointsY/4 && j < 3*numPointsY/4){
						h_height[i + j * (numPointsX)] = 5.5f;
					}
					else{
						h_height[i + j * (numPointsX)] = 1.0f;
					}
				}
			}

		// case 1:
		// 	for(int i = 0; i < numPointsX; i++){
		// 		for(int j = 0 ; j < numPointsY; j++){
		// 			h_height[i + j * (numPointsX)] = 2.0f;
		// 			h_momentumU[i + j * (numPointsX)] = 0.0f;
		// 			h_momentumV[i + j * (numPointsX)] = 0.0f;
					
					
		// 		}
			// }
		}

	cudaMemcpy(d_height, h_height, (numPointsX)*(numPointsY) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_momentumU, h_momentumU, (numPointsX)*(numPointsY) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_momentumV, h_momentumV, (numPointsX)*(numPointsY) * sizeof(float), cudaMemcpyHostToDevice);

}

void SWE::simulate(){
	
	cudaError_t kernelErr;
	dim3 grid(((numPointsX - 1 - (NUM_THREADS_X - 1))/ (NUM_THREADS_X - 2*BOUNDARY_CELL_COUNT)) + 1, ((numPointsY - 1 - (NUM_THREADS_Y - 1))/ (NUM_THREADS_Y - 2*BOUNDARY_CELL_COUNT)) + 1, 1);
	dim3 block(NUM_THREADS_X, NUM_THREADS_Y, 1);

	for(int i = 0; i<NUM_ITERATIONS; i++){
		applySWE <<< grid, block >>> (numPointsX, numPointsY, d_height, d_momentumU, d_momentumV, d_offsetX, d_offsetY, d_height_out, d_momentumU_out, d_momentumV_out);
		
		kernelErr = cudaGetLastError();
		if(kernelErr!=cudaSuccess){
			printf("Error: %s\n", cudaGetErrorString(kernelErr));
		}

		cudaMemcpy(d_height, d_height_out, (numPointsX)*(numPointsY) * sizeof(float), cudaMemcpyDeviceToDevice);


	}
	cudaMemcpy(h_height_out, d_height_out, (numPointsX)*(numPointsY) * sizeof(float), cudaMemcpyDeviceToHost);
}




