find_package(Lua51 REQUIRED)

include_directories(${LUA_INCLUDE_DIR})
add_library(configuration configuration.cc lua_loader.cc)
target_link_libraries(configuration ${LUA_LIBRARIES})