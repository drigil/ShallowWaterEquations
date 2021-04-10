#include <iostream>
#include "SWE.cuh"

int main(){
	int numPointsX = 1024;
	int numPointsY = 1024;
	int conditionNum = 0;

	SWE swe(numPointsX, numPointsY);
	swe.setInitialConditions(conditionNum);
}