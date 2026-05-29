{
  stdenv,
  fetchFromGitHub,
  cmake,
  eigen,
}:
stdenv.mkDerivation rec {
  pname = "poselib";
  version = "2.0.4";

  src = fetchFromGitHub {
    owner = "PoseLib";
    repo = "PoseLib";
    rev = "v${version}";
    hash = "sha256-5cd0k53kqggJCzz3ajPcUeBIi5KuvBUG7SQKsHBWIdU=";
  };
  nativeBuildInputs = [ cmake ];
  buildInputs = [ eigen ];
}
