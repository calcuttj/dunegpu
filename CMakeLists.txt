# ======================================================================
#  dunegpu main build file
#
#  cd .../path/to/build/directory
#  source .../path/to/dunegpu/ups/setup_for_development <-d|-p>
#  cmake [-DCMAKE_INSTALL_PREFIX=/install/path]
#        -DCMAKE_BUILD_TYPE=$CETPKG_TYPE
#        $CETPKG_SOURCE
#  make
#  make test
#  make install
#  make package (builds distribution tarfile)
# ======================================================================


# use cmake 3.12 or later
cmake_minimum_required (VERSION 3.12)

project(dunegpu LANGUAGES CXX C CUDA) ##Note! Need to add CUDA to languages

# cetbuildtools contains our cmake modules
find_package(cetbuildtools REQUIRED)
find_package( CUDA REQUIRED ) ##Note! Make sure to find cuda
find_package(Thrust REQUIRED CONFIG) ##Note! Also finding Thrust
thrust_create_target(Thrust)


list(APPEND CMAKE_MODULE_PATH
     $ENV{CANVAS_ROOT_IO_DIR}/Modules
     $ENV{ART_DIR}/Modules
#     ${CMAKE_CURRENT_SOURCE_DIR}/Modules
)

include(CetCMakeEnv)
cet_cmake_env()


#Note! Creating new macro that only provides these compiler flags
#for a given target. This is important because certain compiler flags
#are incompatible with nvcc
macro(cet_set_target_compiler_flags _target)
  cmake_parse_arguments(CSCF
    "ALLOW_DEPRECATIONS;DWARF_STRICT;ENABLE_ASSERTS;NO_UNDEFINED;WERROR"
    ""
    "DIAGS;DWARF_VER;EXTRA_FLAGS;EXTRA_C_FLAGS;EXTRA_CXX_FLAGS;EXTRA_DEFINITIONS"
    ${ARGN})

  # No non-option arguments.
  if (CSCF_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unexpected extra arguments: ${CSCF_UNPARSED_ARGUMENTS}."
      "\nConsider EXTRA_FLAGS or EXTRA_DEFINITIONS")
  endif()

  # WERROR.
  if (CSCF_WERROR)
    target_compile_options(${_target} PUBLIC -Werror)
  endif()

  # ALLOW_DEPRECATIONS.
  if (CSCF_ALLOW_DEPRECATIONS)
    target_compile_options(${_target} PUBLIC -Wno-error=deprecated-declarations)
  endif()

  # DWARF_VER, DWARF_STRICT.
  if (CSCF_DWARF_VER)
    if (CSCF_DWARF_VER LESS 4)
      message(WARNING "Setting DWARF format version < 4 may impact your"
        " ability to debug modern C++ programs.")
    endif()
    target_compile_options(${_target} PUBLIC -gdwarf-${CSCF_DWARF_VER})
    if (CSCF_DWARF_STRICT)
      target_compile_options(${_target} PUBLIC -gstrict-dwarf)
    endif()
  endif()

  if (CSCF_ENABLE_ASSERTS)
    cet_enable_asserts()
  else()
    cet_maybe_disable_asserts()
  endif()

  # EXTRA_DEFINITIONS.
  #
  # Note that we no longer want the leading "-D", and -U... undefines
  # must be filtered out and handled via target_compile_options() instead.
  list(TRANSFORM CSCF_EXTRA_DEFINITIONS
    REPLACE "^-D([A-Za-z_][A-Za-z_0-9]*)$" "\\1"
    OUTPUT_VARIABLE compile_defs)
  list(FILTER CSCF_EXTRA_DEFINITIONS INCLUDE REGEX "^-U")
  list(FILTER compile_defs EXCLUDE REGEX "^-U")

  target_compile_definitions(${_target} PUBLIC ${compile_defs})
  target_compile_options(${_target} PUBLIC ${CSCF_EXTRA_DEFINITIONS})
  
  # Generally-useful options.
  target_compile_options(${_target} PUBLIC "SHELL:$<$<COMPILE_LANG_AND_ID:$<COMPILE_LANGUAGE>,GNU>:-frecord-gcc-switches>")
  target_compile_options(${_target} PUBLIC "SHELL:$<$<COMPILE_LANG_AND_ID:$<COMPILE_LANGUAGE>,Clang,AppleClang,GNU>:-grecord-gcc-switches>")

  # Add options according to diagnostic mode DIAGS.
  set(diags_vals CAVALIER CAUTIOUS VIGILANT PARANOID)
  string(TOUPPER "${CSCF_DIAGS}" CSCF_DIAGS)
  if (NOT CSCF_DIAGS)
    set(CSCF_DIAGS "CAUTIOUS")
  endif()
  list(FIND diags_vals ${CSCF_DIAGS} diag_idx)
  if (diag_idx GREATER -1)
    message(VERBOSE "selected diagnostic level: ${CSCF_DIAGS}")
    if (diag_idx GREATER 0) # At least CAUTIOUS
      target_compile_options(${_target} PUBLIC "SHELL:$<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Werror=return-type>")
      if (diag_idx GREATER 1) # At least VIGILANT
        target_compile_options(${_target} PUBLIC -Wextra -Wno-long-long -Winit-self)
        target_compile_options(${_target} PUBLIC "SHELL:$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<COMPILE_LANG_AND_ID:$<COMPILE_LANGUAGE>,Clang,AppleClang>,$<AND:$<COMPILE_LANG_AND_ID:$<COMPILE_LANGUAGE>,GNU>,$<VERSION_GREATER_EQUAL:$<$<COMPILE_LANGUAGE>_COMPILER_VERSION>,4.7.0>>>>:-Wno-unused-local-typedefs>")
        target_compile_options(${_target} PUBLIC "SHELL:$<$<COMPILE_LANGUAGE:CXX>:-Wdelete-non-virtual-dtor>") # C++ only
        target_compile_options(${_target} PUBLIC "SHELL:$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:Clang,AppleClang>,$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,4.7.0>>>>:-Woverloaded-virtual -Wnon-virtual-dtor>") # C++ only
        if (diag_idx GREATER 2) # PARANOID
          target_compile_options(${_target} PUBLIC -pedantic
            -Wformat-y2k
            -Wswitch-default
            -Wsync-nand
            -Wtrampolines
            -Wlogical-op
            -Wshadow
            -Wcast-qual)
        endif(diag_idx GREATER 2)
      endif(diag_idx GREATER 1)
    endif(diag_idx GREATER 0)
  else()
    message(FATAL_ERROR "Unrecognized DIAGS option ${CSCF_DIAGS}")
  endif()

  # EXTRA(_C|_CXX)?_FLAGS.
  foreach(lang IN ITEMS C CXX)
    if (CSCF_EXTRA_${lang}_FLAGS)
      warn_deprecated("EXTRA_${lang}_FLAGS" NEW
        "EXTRA_FLAGS with generator expressions")
      list(APPEND CSCF_EXTRA_FLAGS "$<$<COMPILE_LANGUAGE:${lang}>:${CSCF_EXTRA_${lang}_FLAGS}>")
    endif()
  endforeach()
  target_compile_options(${_target} PUBLIC "SHELL:${CSCF_EXTRA_FLAGS}")

  # NO_UNDEFINED.
  if (CSCF_NO_UNDEFINED)
    add_link_options("SHELL:$<IF:$<PLATFORM_ID:Darwin>,LINKER:-undefined$<COMMA>error,LINKER:--unresolved-symbols=ignore-in-shared-libs>")
  else()
    add_link_options("SHELL:$<IF:$<PLATFORM_ID:Darwin>,LINKER:-undefined$<COMMA>dynamic_lookup,LINKER:--unresolved-symbols=ignore-all>")
  endif()
endmacro()


##Note! Turn this off, will replace for each target with the above macro
#cet_set_compiler_flags_cxx(DIAGS CAUTIOUS
#  WERROR
#  NO_UNDEFINED
#  EXTRA_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-pedantic
#  EXTRA_CXX_FLAGS -Wno-unused-local-typedefs
#)

cet_report_compiler_flags()

# these are minimum required versions, not the actual product versions
find_ups_product( larsoft )
find_ups_product( nutools )
find_ups_product( art )
find_ups_product( cetbuildtools ) # LIBRARY_OUTPUT_DIRECTORY, etc.
find_ups_product( larcore )
find_ups_product( larcorealg )
find_ups_product( lardataalg )
find_ups_product( lardataobj )
find_ups_product( larsim )
find_ups_product( dunesw )

# macros for dictionary and simple_plugin
include(ArtDictionary)
include(ArtMake)
include(BuildPlugins)

# add cet_find_library commands here when needed
#find_library(cudartlib cudart)

# ADD SOURCE CODE SUBDIRECTORIES HERE
add_subdirectory(dunegpu)

# tests
add_subdirectory(test)

# ups - table and config files
# must be AFTER all other subdirectories
add_subdirectory(ups)

# packaging utility - must be last
include(UseCPack)
