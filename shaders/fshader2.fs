#version 330 core
in vec4 locationVector;

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
vec3 halfwayVector;

out vec4 FragColor;



void main()

{	
	
    pos_reduced = vec3(locationVector[0], locationVector[1], locationVector[2]);
    
    //to do 
    //Create different fragColor implementations for different cases

    
    //Defining ambient light

    ambientLight = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    ambientLightCoeff = 0.08f;
    ambientLight = ambientLightCoeff*ambientLight; 


    
    //Defining diffuse light
	
	diffLightCoeff = 0.8f;
	lightVector = normalize(pos_reduced - lpos_world);
	positionVector = normalize(worldOrigin - pos_reduced); 
	diffLight = vec4(0.0f, 0.0f, 1.0f, 1.0f);
	diffLight = diffLightCoeff * max(dot(lightVector, positionVector), 0.0f) * diffLight;
	


	//Defining specular light
	
	viewVector = normalize(cameraPosition - pos_reduced);
	halfwayVector = (lightVector + viewVector) / length(lightVector + viewVector);
	specLightCoeff = 0.2f;
	shininessCoeff = 64.0f;
	specLight = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	specLight = specLightCoeff * max(pow(dot(positionVector, halfwayVector), shininessCoeff), 0.0f) * specLight;


	FragColor = diffLight + ambientLight;
};