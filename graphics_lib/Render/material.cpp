#include <glad/glad.h>
#include "material.h"

#include "Utilities/Utils.h"

// #include "Dep/stb/stb_image.h"
// #include "Dep/stb/stb_image_write.h"

material::material() {
}


material::~material() {
}

img_texutre::img_texutre():ogl_tex_id(-1), w(0), h(0), c(0) {
    img.clear();
}

bool img_texutre::read_img(const std::string path) {
    // unsigned char *img_ptr = stbi_load(path.c_str(), &w, &h, &c, 0);
    // if (img_ptr == nullptr) {
    //     WARN("loading iamge: " + path + " failed");
    //     return false;
    // }

    // img.resize(w * h * c, 0xff);
    // std::copy(&img_ptr[0], &img_ptr[w * h * c -1], std::back_inserter(img));
    // stbi_image_free(img_ptr);

    // // loading to opengl
    // glGenTextures(1, &ogl_tex_id);
    // glBindTexture(GL_TEXTURE_2D, ogl_tex_id);
    // if (c == 4)
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, img.data());
    // else if (c==3)
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data());
    // else {
    //     WARN("unkonwn channel: " + std::to_string(c));
    // }
    // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    // glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    return true;
}