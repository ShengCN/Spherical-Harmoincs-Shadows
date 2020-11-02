#version 430

flat in int vs_vid;
in vec3 vs_pos;
in vec3 vs_color;
in vec3 vs_norm;
in vec2 vs_uvs;

uniform float slider;
uniform int cur_selected_id;
uniform int is_draw_normal;
uniform sampler2D sh_texture;
uniform sampler2D sh_light;
uniform int sh_w;
uniform int sh_h;

out vec4 frag_color;

float lerp(float t1, float t2, float fract){
    return (1.0 - fract) * t1 + fract * t2;
}

void main(){
    float z1 = -100.0, z2 = 100.0;
    float z = lerp(z1, z2, slider);

    vec3 light = vec3(0.0, 100.0, 80.0);
    float ka = 0.3f;
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

    float v = vs_vid * 1.0f/sh_h;

    // if (sh_w > 1)
    
    col = vec3(0.0);
    for(int i = 0; i < sh_w; ++i) {
        float u = (i + 0.5) / sh_w;
        float coeff = texture(sh_texture, vec2(u,v)).r;
        float light_coeff = texture(sh_light, vec2(u,0)).r;
        col += light_coeff * coeff * vs_color;
    }
    
    frag_color = vec4(col, 1.0f);
}
