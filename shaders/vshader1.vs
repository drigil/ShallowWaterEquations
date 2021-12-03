#version 330 core
layout (location = 0)in vec3 vVertex;
layout (location = 1)in vec3 vertex_norm;

uniform mat4 vModel;
uniform mat4 vView;
uniform mat4 vProjection;

vec3 eyepos;
vec3 eye_normal;

vec3 lpos_world = vec3(0.0f, 100.0f, 1000.0f);
vec3 worldOrigin = vec3(0.0f, 0.0f, 0.0f);
vec3 cameraPosition  = vec3(0.0, 0.0, 1.0);

vec4 ambientLight;
float ambientLightCoeff;

vec4 diffLight;
float diffLightCoeff;

vec4 specLight;
float specLightCoeff;
float shininessCoeff;

vec3 pos_reduced;

vec3 lightVector;
vec3 positionVector;
vec3 viewVector;
vec3 reflectedVector;

out vec4 fragColor;

void main(){

    gl_Position = vProjection * vView * vModel * vec4(vVertex, 1.0);
    pos_reduced = vec3(gl_Position[0], gl_Position[1], gl_Position[2]);
    
    //to do 
    //Create different fragColor implementations for different cases

    
    //Defining ambient light

    ambientLight = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    ambientLightCoeff = 0.3f;
    ambientLight = ambientLightCoeff*ambientLight; 


    
    //Defining diffuse light
	
	diffLightCoeff = 0.8f;
	lightVector = normalize(pos_reduced - lpos_world);
	positionVector = normalize(worldOrigin - pos_reduced); 
	diffLight = vec4(0.0f, 0.0f, 1.0f, 1.0f);
	diffLight = diffLightCoeff * max(dot(lightVector, positionVector), 0.0f) * diffLight;
	


	//Defining specular light

	reflectedVector = (2 * dot(lightVector, positionVector) * positionVector) - lightVector;
	reflectedVector = normalize(reflectedVector);
	viewVector = normalize(cameraPosition - pos_reduced);
	specLightCoeff = 0.2f;
	shininessCoeff = 64.0f;
	specLight = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	specLight = specLightCoeff * max(pow(dot(reflectedVector,viewVector), shininessCoeff), 0.0f) * specLight;


	fragColor = diffLight; // + ambientLight + specLight;
	//fragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);

}