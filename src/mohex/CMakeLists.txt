file(GLOB mohex_SRC *.cpp *.hpp)
include_directories(${top_srcdir}/src
        ${top_srcdir}/src/book
        ${top_srcdir}/src/hex
        ${top_srcdir}/src/util
        ${top_srcdir}/src/commonengine
        ${top_srcdir}/src/simpleplayers
        ${top_srcdir}/src/solver
        ${top_srcdir}/src/smartgame
        ${top_srcdir}/src/gtpengine
        ${top_srcdir}/src/neuralnet
        )
IF (APPLE)
    MESSAGE(STATUS "Mac OS, change the link dir to berkeley-db if necessary")
    link_directories(/usr/local/Cellar/berkeley-db/6.2.23/lib)
ELSEIF (UNIX)
    MESSAGE(STATUS "Ubuntu, link /usr/lib/x86_64-linux-gnu")
    link_directories(/usr/lib/x86_64-linux-gnu)
ENDIF()

add_executable(mohexGNN ${mohex_SRC})
target_link_libraries(mohexGNN
        nnevaluate
        hex
        util 
        book 
        fuego_smartgame
        fuego_gtpengine
        solver
        commonengine
        simpleplayers
        pthread
        db
        ${Boost_LIBRARIES}
        )
#boost_system boost_filesystem boost_program_options boost_thread db)
