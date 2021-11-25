configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(CIRCT_HLS_TEST_DEPENDS
  FileCheck count not
  hls-opt
  )

add_lit_testsuite(check-circt-hls "Running the CIRCT HLS regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_HLS_TEST_DEPENDS}
  )
set_target_properties(check-circt-hls PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIRCT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_HLS_TEST_DEPS}
)
