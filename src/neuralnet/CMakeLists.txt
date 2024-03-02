file(GLOB nnevaluate_SRC *.cpp *.h *.hpp)

include_directories(
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util)

find_package(Torch REQUIRED)
add_library(nnevaluate STATIC ${nnevaluate_SRC})

target_link_libraries(
    nnevaluate
    pthread
    "${TORCH_LIBRARIES}"
)

set_property(TARGET nnevaluate PROPERTY CXX_STANDARD 20)
