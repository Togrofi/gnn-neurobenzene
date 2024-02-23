file(GLOB nnevaluate_SRC *.cpp *.h *.hpp)

include_directories(
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util)

add_library(nnevaluate STATIC ${nnevaluate_SRC})

target_link_libraries(
    nnevaluate
    pthread
)
