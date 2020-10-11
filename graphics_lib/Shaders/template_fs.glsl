#version 430

in vec3 vs_pos;
in vec3 vs_color;
in vec3 vs_norm;
in vec2 vs_uvs;

uniform float slider;
uniform int cur_selected_id;
uniform int is_draw_normal;

out vec4 frag_color;

float lerp(float t1, float t2, float fract){
    return (1.0 - fract) * t1 + fract * t2;
}

void main(){
    float z1 = -100.0, z2 = 100.0;
    float z = lerp(z1, z2, slider);

    vec3 light = vec3(0.0, 100.0, 80.0);
    float ka = 0.3;
    float kd = clamp(dot(normalize(light - vs_pos),vs_norm), 0.0f, 1.0f);

    vec3 col = (ka + (1.0 - ka) * kd) * vs_color;
    // col = vs_norm;

    if(cur_selected_id == 1){
        float fract = 0.8;
        col = (1.0 - fract) * col + fract * vec3(1.0, 0.0, 0.0);
    }
        
    if(is_draw_normal > 0){
        col = vs_norm;
    }

    frag_color = vec4(col, 0.5);
}