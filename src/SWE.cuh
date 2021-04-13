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

		// height and momentum variables
		float *h_height_out, *h_momentumU_out, *h_momentumV_out; //Host
		float *d_height_out, *d_momentumU_out, *d_momentumV_out; // Device

		// Offsets for boundary conditions
		int *h_offsetX, *h_offsetY;
		int *d_offsetX, *d_offsetY;
	
	public:
		SWE(int numPointsX, int numPointsY);	
		void setInitialConditions(int conditionNum);
		void simulate();

};

#endif