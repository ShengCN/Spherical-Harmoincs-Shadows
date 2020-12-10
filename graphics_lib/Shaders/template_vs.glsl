#version 430

layout(location=0) in vec3 pos_attr;
layout(location=1) in vec3 norm_attr;
layout(location=2) in vec3 col_attr;
layout(location=3) in vec2 uv_attr;

uniform mat4 PVM;
uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

out vec3 vs_pos;  // world space position
out vec3 vs_color;
out vec3 vs_norm;
out vec2 vs_uvs;

void main(){
    gl_Position = PVM * vec4(pos_attr,1.0f);

    // pass values to next step
    vs_pos = vec3(M * vec4(pos_attr, 1.0));
    vs_color = col_attr;
    vs_norm = normalize((transpose(inverse(M)) * vec4(norm_attr, 0.0f)).xyz);
    vs_uvs = uv_attr;    
}