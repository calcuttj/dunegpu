include_directories("${CUDA_INCLUDE_DIRS}")
add_library(gpuadd SHARED
  add.cu
  add.cuh
  SimEnergyDepCuda.cu
  SimEnergyDepCuda.cuh

  )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(gpuadd
                        PUBLIC
                        cxx_std_17
                        )

#target_compile_options(gpuadd
#                       PUBLIC
#                       $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60
#                       )

#message("Compile options!!")
#get_target_property(gpuadd_compile_options gpuadd COMPILE_OPTIONS)
#message(${gpuadd_compile_options})

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( gpuadd
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

target_link_libraries(gpuadd PUBLIC lardataobj::Simulation  ${CUDA_LIBRARIES})

cet_build_plugin(GPUAnalyzer art::EDAnalyzer
  LIBRARIES PRIVATE
  gpuadd
  #lardataobj::Simulation
  #CUDA::cudart
  #cuda
  art::Framework_Principal
  art::Framework_Services_Registry
  art_plugin_support::toolMaker
  canvas::canvas
  fhiclcpp::fhiclcpp
  CLHEP::Random
)



cet_set_target_compiler_flags(dunegpu_Base_GPUAnalyzer_module)
function(get_all_targets var)
    set(targets)
    get_all_targets_recursive(targets ${CMAKE_CURRENT_SOURCE_DIR})
    set(${var} ${targets} PARENT_SCOPE)
endfunction()

macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

get_all_targets(all_targets)
message("All targets: ${all_targets}")



install(TARGETS gpuadd DESTINATION slf7.x86_64.e20.prof/lib/)

install_headers()
install_fhicl()
install_source()
install_scripts()

