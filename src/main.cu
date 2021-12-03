#include <iostream>
#include "SWE.cuh"
#include "utils.h"
#include <unistd.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

int width = 640, height=640; // Window dimensions

int main(){
	int numPointsX = 80;
	int numPointsY = 56;
	int conditionNum = 0;

	SWE swe(numPointsX, numPointsY);
	swe.setInitialConditions(conditionNum);
	
	// Setup window
    GLFWwindow *window = setupWindow(width, height);
    ImGuiIO& io = ImGui::GetIO(); // Create IO object

    ImVec4 clearColor = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

    //Phong Shading
    unsigned int shaderProgram = createProgram("../shaders/vshader2.vs", "../shaders/fshader2.fs");
    
    //Gouraud Shading
    // unsigned int shaderProgram = createProgram("../shaders/vshader1.vs", "../shaders/fshader.fs");
    glUseProgram(shaderProgram);

    setupModelTransformation(shaderProgram);
    setupViewTransformation(shaderProgram);
    setupProjectionTransformation(shaderProgram, width , height);
    
    int vVertex_attrib = glGetAttribLocation(shaderProgram, "vVertex");
    if(vVertex_attrib == -1) {
        std::cout << "Could not bind location: vVertex\n" ;
        exit(0);
    }else{
        std::cout << "vVertex found at location " << vVertex_attrib << std::endl;
    }

	// Construct 1D vertex array to display

    int numVertices = (numPointsX-1) * (numPointsY-1) * 2 * 9; // 3 coordinates per vertex of all triangles
    GLfloat glVertices[numVertices];

    glUseProgram(shaderProgram);

  	int iterationNum = 0;
    while (!glfwWindowShouldClose(window) && (iterationNum < NUM_ITERATIONS))
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        glUseProgram(shaderProgram);
        // showOptionsDialog(shaderProgram, angle, matrix);

        {
            // static float f = 0.0f;
            // static int counter = 0;

            ImGui::Begin("Information");                          
            ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // Computations
        swe.simulate();
		cudaError_t kernelErr = cudaGetLastError();
		if(kernelErr!=cudaSuccess){
			printf("Error: %s\n", cudaGetErrorString(kernelErr));
		}

		// // Debugging
		// for(int i = 0; i<numPointsX - 1; i++){
		// 	for(int j = 0; j<numPointsY - 1; j++){
		// 		printf("%f ", swe.h_height_out[i + j * (numPointsX)]);
		// 	}
		// 	printf("\n");
		// }

	    int counter = 0;

	    for(int i = 0; i<numPointsX - 1; i++){
			for(int j = 0; j<numPointsY - 1; j++){
				glVertices[counter] = i;
		        glVertices[counter + 1] = swe.h_height_out[i + j * (numPointsX)];
		        glVertices[counter + 2] = j;

		        counter = counter + 3;

		        glVertices[counter] = i + 1;
		        glVertices[counter + 1] = swe.h_height_out[i + 1 + j * (numPointsX)];
		        glVertices[counter + 2] = j;

		        counter = counter + 3;
		        
		        glVertices[counter] = i;
		        glVertices[counter + 1] = swe.h_height_out[i + (j + 1) * (numPointsX)];
		        glVertices[counter + 2] = j + 1;

		        counter = counter + 3;
		        
		        glVertices[counter] = i + 1;
		        glVertices[counter + 1] = swe.h_height_out[i + 1 + j * (numPointsX)];
		        glVertices[counter + 2] = j;

		        counter = counter + 3;
		        
		        glVertices[counter] = i + 1;
		        glVertices[counter + 1] = swe.h_height_out[i + 1 + (j + 1) * (numPointsX)];
		        glVertices[counter + 2] = j + 1;

		        counter = counter + 3;
		        
		        glVertices[counter] = i;
		        glVertices[counter + 1] = swe.h_height_out[i + (j  + 1) * (numPointsX)];
		        glVertices[counter + 2] = j + 1;

		        counter = counter + 3;

			}
		}

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        GLuint vertex_VBO;
	    glGenBuffers(1, &vertex_VBO);
	    glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO);
	    glBufferData(GL_ARRAY_BUFFER, sizeof(glVertices), glVertices, GL_STATIC_DRAW); //else tri_points
	      
	    
	    GLuint obj_VAO;
	    glGenVertexArrays(1, &obj_VAO);
	    glBindVertexArray(obj_VAO);
	    glBindBuffer(GL_ARRAY_BUFFER, vertex_VBO);
	    glVertexAttribPointer(static_cast<uint>(vVertex_attrib), 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	    glEnableVertexAttribArray(static_cast<uint>(vVertex_attrib));


        glBindVertexArray(obj_VAO); 

        // glDrawArrays(GL_POINTS, 0, numVertices);//else tri_points   GL_TRIANGLES
        glDrawArrays(GL_TRIANGLES, 0, numVertices);//else tri_points   GL_TRIANGLES

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        iterationNum++;
        printf("Iteration Num %d\n", iterationNum);
        
        // usleep(1000);
    }

    cleanup(window);

    return 0;
	
}