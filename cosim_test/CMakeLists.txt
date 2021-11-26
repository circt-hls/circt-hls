configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INTEGRATION_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CIRCT_HLS_EXTENDED_INTEGRATION_TEST_DEPENDS
  FileCheck count not split-file
  hls-opt
  circt-hls-testscripts
)

add_lit_testsuite(check-circt-hls-integration-extended "Running the CIRCT HLS extended integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_HLS_EXTENDED_INTEGRATION_TEST_DEPENDS}
  )
set_target_properties(check-circt-hls-integration-extended PROPERTIES FOLDER "ExtendedIntegrationTests")

add_lit_testsuites(CIRCT_HLS_EXTENDED_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_HLS_EXTENDED_INTEGRATION_TEST_DEPS}
)
