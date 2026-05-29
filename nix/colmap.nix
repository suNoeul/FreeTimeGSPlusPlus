{
  lib,
  fetchFromGitHub,
  cmake,
  boost,
  ceres-solver,
  eigen,
  freeimage,
  glog,
  libGLU,
  glew,
  flann,
  cgal,
  gmp,
  mpfr,
  faiss,
  autoAddDriverRunpath,
  config,
  stdenv,
  sqlite,
  callPackage,
  cudaSupport ? config.cudaSupport,
  cudaPackages,
  cudaCapabilities ? cudaPackages.flags.cudaCapabilities,
  cudaArchitectures ? lib.strings.concatStringsSep ";" (
    map cudaPackages.flags.dropDots cudaCapabilities
  ),
}:

assert cudaSupport -> cudaPackages != { };

let
  boost_static = boost.override { enableStatic = true; };
  stdenv' = if cudaSupport then cudaPackages.backendStdenv else stdenv;
  poselib = callPackage ./poselib.nix { };

  # TODO: migrate to redist packages
  inherit (cudaPackages) cudatoolkit;
in
stdenv'.mkDerivation rec {
  version = "3.12.3";
  pname = "colmap";
  src = fetchFromGitHub {
    owner = "colmap";
    repo = "colmap";
    rev = version;
    hash = "sha256-cxUEHcEZnprapnywxW181BR3iNmxlA7CbsK8T3eFqlA=";
  };

  cmakeFlags = lib.optionals cudaSupport [
    (lib.cmakeBool "CUDA_ENABLED" true)
    (lib.cmakeFeature "CMAKE_CUDA_ARCHITECTURES" cudaArchitectures)
    (lib.cmakeBool "GUI_ENABLED" false)
    (lib.cmakeBool "FETCH_POSELIB" false)
    (lib.cmakeBool "FETCH_FAISS" false)
  ];

  buildInputs =
    [
      boost_static
      ceres-solver
      eigen
      freeimage
      glog
      libGLU
      glew
      sqlite
      flann
      cgal
      gmp
      mpfr
      poselib
      faiss
    ]
    ++ lib.optionals cudaSupport [
      cudatoolkit
      cudaPackages.cuda_cudart.static
    ];

  nativeBuildInputs =
    [ cmake ]
    ++ lib.optionals cudaSupport [
      autoAddDriverRunpath
    ];
}
