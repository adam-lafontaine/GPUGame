#include "image_types.hpp"


namespace libimage
{
    void read_image_from_file(const char* img_path_src, Image& image_dst);

    void resize_image(Image const& image_src, Image& image_dst);

    void make_image(Image& image_dst, u32 width, u32 height);

    inline void destroy_image(Image& image) { if(image.data) { free(image.data); } }
}

