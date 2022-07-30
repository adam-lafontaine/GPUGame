#include "../utils/types.hpp"


namespace libimage
{
    void read_image_from_file(const char* img_path_src, image_t& image_dst);

    void resize_image(image_t const& image_src, image_t& image_dst);

    void make_image(image_t& image_dst, u32 width, u32 height);

    inline void destroy_image(image_t& image) { if(image.data) { free(image.data); } }
}

