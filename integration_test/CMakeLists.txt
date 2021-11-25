configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INTEGRATION_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CIRCT_HLS_INTEGRATION_TEST_DEPENDS
  FileCheck count not split-file
  hls-opt)

add_lit_testsuite(check-circt-hls-integration "Running the CIRCT HLS integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_HLS_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-circt-hls-integration PROPERTIES FOLDER "IntegrationTests")

add_lit_testsuites(CIRCT_HLS_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_HLS_INTEGRATION_TEST_DEPS}
)
