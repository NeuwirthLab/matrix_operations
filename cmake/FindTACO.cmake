find_path(
  TACO_INCLUDE_DIR
  NAMES "taco.h"
  PATH_SUFFIXES "include"
)

find_library(
  TACO_LIBRARY
  NAMES libtaco.so
  PATH_SUFFIXES "lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	TACO REQUIRED_VARS TACO_INCLUDE_DIR TACO_LIBRARY
)

if(TACO_FOUND)
  if(NOT TARGET TACO::TACO)
	  add_library(TACO::TACO UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
      TACO::TACO PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${TACO_INCLUDE_DIR}
                            IMPORTED_LOCATION ${TACO_LIBRARY}
    )
  endif()
endif()
mark_as_advanced(TACO_INCLUDE_DIR TACO_LIBRARY)
