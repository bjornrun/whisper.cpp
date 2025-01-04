find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED 
    gstreamer-1.0
    gstreamer-base-1.0
    gstreamer-audio-1.0
)

message(STATUS "GST_LIBRARY_DIRS: ${GST_LIBRARY_DIRS}")
message(STATUS "GST_INCLUDE_DIRS: ${GST_INCLUDE_DIRS}")

set(TARGET gstwhispersink)

add_library(${TARGET} SHARED
    whisper_sink.cpp
    whisper_sink.h
)

include(DefaultTargetOptions)

# Fix the include directories - use the correct variables from pkg-config
target_include_directories(${TARGET} PRIVATE
    ${GST_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PRIVATE
    whisper
    common
    ${GST_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# Set the output directory for the plugin
set_target_properties(${TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/gstreamer-1.0"
    PREFIX ""
)

message(STATUS "Plugin will be built to: ${CMAKE_BINARY_DIR}/lib/gstreamer-1.0")

install(TARGETS ${TARGET}
    LIBRARY DESTINATION lib/gstreamer-1.0
) 