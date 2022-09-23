project(
    'pharos',
    'cpp',
    version : '1.0.0',
)

src = [
    'main.cpp'  
]

deps = [
    dependency('opencv4'),
    dependency('apriltag')
]

build_args = [
]


build_args += [
  '-DPROJECT_NAME=' + meson.project_name(),
  '-DPROJECT_VERSION=' + meson.project_version(),
]


project_target = executable(
  meson.project_name(),
  src,
  dependencies: deps,
  cpp_args : build_args,
)
