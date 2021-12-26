  # The cosim tests contain Verilator builds which use full parallelism; these builds
  # take up the majority of the total time of each test. As such, keep lit parallelism
  # at a minimum.
  # @TODO: this may be increase by a few threads if you have a very powerful machine...
include(ProcessorCount)
ProcessorCount(NCores)
math(EXPR NJobs "${NCores}/4")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  INTEGRATION_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CIRCT_HLS_EXTENDED_INTEGRATION_TEST_DEPENDS
  FileCheck count not split-file
  hls-opt
  hlstool
)

add_lit_testsuite(check-circt-hls-cosim "Running the CIRCT HLS cosimulation test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_HLS_EXTENDED_INTEGRATION_TEST_DEPENDS}
  ARGS -j${NJobs}
  )
set_target_properties(check-circt-hls-cosim PROPERTIES FOLDER "ExtendedIntegrationTests")

add_lit_testsuites(CIRCT_HLS_EXTENDED_INTEGRATION ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CIRCT_HLS_EXTENDED_INTEGRATION_TEST_DEPS}
)
