#include "SWE.cuh"

// To Do - 
// 1. Use cuda's math.h for speedup
// 2. Check if shared memory overlaps not causing issues - Done
// 3. Check bcCount
// 4. Matmul and some matrices left - Done
// 5. Deal with shared memory out of bounds access - Done
// 6. Global index - Done
// 7. Offsets

// Errors Occurred
// 1. Too many resources - Solved by reducing block dimensions
// 2. Warp out of bound - Solved (Hopefully)


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
__global__ void applySWE(int numPointsX, int numPointsY, float* d_height, float* d_momentumU, float* d_momentumV, int* d_offsetX, int* d_offsetY){

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

	int x = blockIdx.x * (NUM_THREADS_X - 2 * BOUNDARY_CELL_COUNT) + threadIdx.x;
	int y = blockIdx.y * (NUM_THREADS_Y - 2 * BOUNDARY_CELL_COUNT) + threadIdx.y;

	int localX = threadIdx.x + 2;
	int localY = threadIdx.y + 2;

	float height = d_height[x + y*gridDim.x*blockDim.x];
	float momentumU = (height != 0.0) ? d_momentumU[x + y*gridDim.x*blockDim.x] / height : 0.0; 
	float momentumV = (height != 0.0) ? d_momentumV[x + y*gridDim.x*blockDim.x] / height : 0.0;

	float3 uCurr = make_float3(height, momentumU * height, momentumV * height);

	int offsetX = d_offsetX[x + y*gridDim.x*blockDim.x];
	int offsetY = d_offsetY[x + y*gridDim.x*blockDim.x];

	// Used to compute alpha
	float eigenX = abs(momentumU) + sqrt(GRAVITY * height);
	float eigenY = abs(momentumV) + sqrt(GRAVITY * height);

	float bathymetryVal = 1.0f; // Implement later using height map

	terrainArr[localY][localX] = bathymetryVal;
	pointInfoArr[localY][localX] = make_float3(height, eigenX, eigenY);

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
	float3 dfduR1 = make_float3(0.0f, -momentumU * momentumU + GRAVITY * height, -momentumU * momentumV);
	float3 dfduR2 = make_float3(1.0f, 2.0f * momentumU,  momentumV);
	float3 dfduR3 = make_float3(0.0f, 0.0f, momentumU);

	float3 dgduR1 = make_float3(0.0f, -momentumU * momentumV, -momentumV * momentumV + GRAVITY * height);
	float3 dgduR2 = make_float3(0.0f, momentumV, 0.0f);
	float3 dgduR3 = make_float3(1.0f, momentumU, 2.0f * momentumV);
	
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
	// float3 sourceLow = make_float3(1.0f, 1.0f, 1.0f);

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
	fluxGArr[localY][localX] = make_float3(height + bathymetryVal, momentumU, momentumV); // Reuse shared memory to store point set quantities again

	__syncthreads(); // Fill shared memory with point information and F Flux values for X direction reconstruction

	// Set flux boundaries
	if((localX + offsetX >= 0) && (localX + offsetX < NUM_THREADS_X)){
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
	float3 fNegX_iN_j =   0.5 * (fluxFArr[localY][localX + 1]     + alphaX * fluxGArr[localY][localX + 1]    );
	float3 fNegX_iNN_j = 0.5 * (fluxFArr[localY][localX + 2] + alphaX * fluxGArr[localY][localX + 2]); //iN dentoes i next

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
	if (((localY + offsetY) >= 0) && ((localY + offsetY) < NUM_THREADS_Y)) {
		pointInfoArr[localY + offsetY][localX] = gTilde * make_float3(-1.0f, 1.0f, 1.0f);
	}

	__syncthreads();


	float3 outFluxPosY, outFluxNegY;
	float outSourcePosY, outSourceNegY;

	// Applying Lax Friedrich flux splitting
	// Positive flux part
	float3 fPosY_i_jP = 0.5 * (fluxFArr[localY - 1][localX] + alphaY * fluxGArr[localY - 1][localX]); // iP denotes i previous
	float3 fPosY_i_j =   0.5 * (fluxFArr[localY][localX]     + alphaY * fluxGArr[localY][localX]);
	float3 fPosY_i_jN = 0.5 * (fluxFArr[localY + 1][localX] + alphaY * fluxGArr[localY + 1][localX]); //iN dentoes i next

	// Reconstruct positive X
	WENOPosY(fPosY_i_jP, fPosY_i_j, fPosY_i_jN, terrainArr[localY - 1][localX], terrainArr[localY][localX], terrainArr[localY + 1][localX], outFluxPosY, outSourcePosY);

	// Negative flux part
	float3 fNegY_i_j = 0.5 * (fluxFArr[localY][localX] - alphaY * fluxGArr[localY][localX]); // iP denotes i previous
	float3 fNegY_i_jN =   0.5 * (fluxFArr[localY + 1][localX] + alphaY * fluxGArr[localY + 1][localX]);
	float3 fNegY_i_jNN = 0.5 * (fluxFArr[localY + 2][localX] + alphaY * fluxGArr[localY + 2][localX]); //iN dentoes i next

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

	float3 uNew = uCurr - ((dt / dx) * (finalFluxX - fluxFArr[localY][localX - 1])) - ((dt/dy) * (finalFluxY -fluxGArr[localY - 1][localX])) - (GRAVITY * (height + bathymetryVal) * make_float3(0.0, (dt/dx) * (outSourceX - terrainArr[localY][localX - 1]), (dt/dy) * (outSourceY - pointInfoArr[localY - 1][localX].z)));

	// Find the veloctities for the current cell
	float heightTerm = uNew.x * uNew.x * uNew.x * uNew.x;

	// Desinguralize velocities
	float momentumUTerm = (uNew.x < DESING_EPSILON) ? sqrt(2.0f) * uNew.x * uNew.y / sqrt(heightTerm + max(heightTerm, DESING_EPSILON)):uNew.y / uNew.x;
	float momentumVTerm = (uNew.x < DESING_EPSILON) ? sqrt(2.0f) * uNew.x * uNew.z / sqrt(heightTerm + max(heightTerm, DESING_EPSILON)):uNew.z / uNew.x;

	// Consistency requirement
	uNew.x = max(0.0, uNew.x);
	uNew.y = uNew.x * momentumUTerm;
	uNew.z = uNew.x * momentumVTerm;

	// Check if inner domain cell
	bool isDomainInner = (x >= BOUNDARY_CELL_COUNT) && (x<(numPointsX + BOUNDARY_CELL_COUNT)) && (y >= BOUNDARY_CELL_COUNT) && (y < (numPointsY + BOUNDARY_CELL_COUNT));

	// Check if in inner domain of current patch
	bool isPatchInner = (localX >= BOUNDARY_CELL_COUNT) && (localX < (NUM_THREADS_X - BOUNDARY_CELL_COUNT)) && (localY >= BOUNDARY_CELL_COUNT) && (localY < (NUM_THREADS_Y - BOUNDARY_CELL_COUNT));

	if(isDomainInner && isPatchInner){
		d_height[x + y*gridDim.x*blockDim.x] = uNew.x;
		d_momentumU[x + y*gridDim.x*blockDim.x] = uNew.y;
		d_momentumV[x + y*gridDim.x*blockDim.x] = uNew.z;
		// Assign offsets too
	}

	// Set the boundary conditions
	d_height[x + offsetX + y*gridDim.x*blockDim.x] = uNew.x;
	d_momentumU[x + offsetX + y*gridDim.x*blockDim.x] = -uNew.y;
	d_momentumV[x + offsetX + y*gridDim.x*blockDim.x] = -uNew.z;

	d_height[x + (y + offsetY)*gridDim.x*blockDim.x] = uNew.x;
	d_momentumU[x + (y + offsetY)*gridDim.x*blockDim.x] = -uNew.y;
	d_momentumV[x + (y + offsetY)*gridDim.x*blockDim.x] = -uNew.z;

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

	cudaMemcpy(d_height, h_height, (numPointsX + 2)*(numPointsY + 2) * sizeof(float), cudaMemcpyHostToDevice);

}

void SWE::simulate(){
	
	cudaError_t kernelErr;
	
	for(int i = 0; i<NUM_ITERATIONS; i++){
		applySWE <<< dim3(numPointsX / NUM_THREADS_X, numPointsY / NUM_THREADS_Y, 1), dim3(NUM_THREADS_X, NUM_THREADS_Y, 1) >>> (numPointsX, numPointsY, d_height, d_momentumU, d_momentumV, d_offsetX, d_offsetY);
		
		kernelErr = cudaGetLastError();
		if(kernelErr!=cudaSuccess){
			printf("Error: %s\n", cudaGetErrorString(kernelErr));
		}

	}
	cudaMemcpy(h_height, d_height, (numPointsX + 2)*(numPointsY + 2) * sizeof(float), cudaMemcpyDeviceToHost);
}




