// Node-side stand-ins for the env.host_* imports declared in runtime/os/os_shim.cpp.
// Linked into the wasm64 test executables by tools/run_wasm_tests.py (cmake option
// PLAYGTA6_WASM_TESTS); the browser build gets the real implementations from the page glue.
addToLibrary({
  host_log__deps: ['$UTF8ToString'],
  host_log: (ptr, len) => {
    out('[host_log] ' + UTF8ToString(Number(ptr), Number(len)));
  },
  // HTTPFS: no host filesystem in tests; every open fails like a missing file.
  host_httpfs_open: (ptr, len) => -1,
  host_httpfs_read: (handle, offset, dst, len) => -1n,
  host_httpfs_close: (handle) => {},
  host_wgpu_submit: (ptr, len) => {},
});
