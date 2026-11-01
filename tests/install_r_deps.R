# Install the R daarem package. The parity tests compare pydaarem with it.
version <- "0.7"
if (!requireNamespace("daarem", quietly = TRUE) ||
    packageVersion("daarem") != version) {
  install.packages("daarem", repos = "https://cloud.r-project.org")
}
stopifnot(packageVersion("daarem") == version)
