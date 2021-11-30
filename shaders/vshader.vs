#version 330 core
layout (location = 0) in vec3 vVertex;
//layout (location = 1) in vec3 aNormal;

uniform mat4 vModel;
uniform mat4 vView;
uniform mat4 vProjection;



void main()
{
 
  //gl_Position = vec4(vVertex.x, vVertex.y, vVertex.z, 1.0);
  gl_Position = vProjection*vView*vModel*vec4(vVertex.x, vVertex.y, vVertex.z, 1.0);
  
 
}