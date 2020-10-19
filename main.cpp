#include "otb_window.h"

using namespace glm;

int w = 1720, h = 1280;
int main(int argc, char* argv[]) {

    otb_window wnd;
    wnd.create_window(w, h,"OTB");
    wnd.show();

	return 0;
}