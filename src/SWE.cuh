#ifndef __SWE_H__
#define __SWE_H__

#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_functions.h"
#include "cuda_helper_functions.cuh"
#include "defines.h"

class SWE{

	public:
		// Number of points in x axis and y axis
		int numPointsX, numPointsY;

		// height and momentum variables
		float *h_height, *h_momentumU, *h_momentumV; //Host
		float *d_height, *d_momentumU, *d_momentumV; // Device

		// horizontal and vertical flux
		// F and G are fluxes along X and Y direction
		// h, hu and hv are the terms of F and G fluxes
		float *h_Fh, *h_Fhu, *h_Fhv, *h_Gh, *h_Ghu, *h_Ghv;  
		float *d_Fh, *d_Fhu, *d_Fhv, *d_Gh, *d_Ghu, *d_Ghv;  

		// max height, max velocity, characteristic velocity
		float *h_maxHeight, *h_maxVelocity, *h_characteristicVelocity;
		float *d_maxHeight, *d_maxVelocity, *d_characteristicVelocity;

		// Offsets for boundary conditions
		int *h_offsetX, *h_offsetY;
		int *d_offsetX, *d_offsetY;
	
	public:
		SWE(int numPointsX, int numPointsY);	
		void setInitialConditions(int conditionNum);
		void simulate();

};

#endif