load_stuff <- function() {
  for (package in c("tidyverse", "docstring")) {
    if (!require(package, character.only = TRUE)) { 
      install.packages(package)
      library(package, character.only = TRUE)
    }
  }
  install_sva <- function() {
    ## try http:// if https:// URLs are not supported
    source("https://bioconductor.org/biocLite.R")
    biocLite("sva")
  }
  if (!require("sva")) install_sva()
  library(sva)
}
suppressMessages(load_stuff())

is.whole <- function(a, tol = 1e-7) { 
  # Snatched from https://stat.ethz.ch/pipermail/r-help/2003-April/032471.html
  is.eq <- function(x,y) { 
    r <- all.equal(x,y, tol=tol)
    is.logical(r) && r 
  }
  (is.numeric(a) && is.eq(a, floor(a))) ||
    (is.complex(a) && {ri <- c(Re(a),Im(a)); is.eq(ri, floor(ri))})
}

varying_row_mask <- function(matrix_)
{
  #' Get a varying row mask
  #' 
  #' Get a boolean vector mask for the rows with & without varying values.
  #'
  #' @param matrix_ A numerical matrix.
  #' 
  #' @return Boolean vector.
  #'
  #' @examples
  #' x <- matrix(c(2, 1, 3, 3, 1, 4, 4, 1, 5, 5, 1, 6), nrow=3, ncol=4)
  #' x
  #' ##      [,1] [,2] [,3] [,4]
  #' ## [1,]    2    3    4    5
  #' ## [2,]    1    1    1    1
  #' ## [3,]    3    4    5    6
  #' varying_row_mask(x)
  #' ## [1]  TRUE FALSE  TRUE
  matrix_ %>%
    tbl_df() %>%
    mutate_(min = min(names(.)), max = max(names(.))) %>%
    mutate(nonvarying = max != min) %>%
    .$nonvarying
}

remove_nonvarying_rows <- function(matrix_)
{
  #' Remove features that don't vary.
  #'
  #' ComBat requires that all inputs have some variance, so we need to remove
  #' any nonvarying rows before passing a matrix into ComBat.
  #'
  #' @param matrix_ A numerical matrix.
  #'
  #' @return The original matrix without any nonvarying rows.
  #' 
  #' @examples
  #' x <- matrix(c(2, 1, 3, 3, 1, 4, 4, 1, 5, 5, 1, 6), nrow=3, ncol=4)
  #' x
  #' ##      [,1] [,2] [,3] [,4]
  #' ## [1,]    2    3    4    5
  #' ## [2,]    1    1    1    1
  #' ## [3,]    3    4    5    6
  #' remove_nonvarying_rows(x)
  #' ##      [,1] [,2] [,3] [,4]
  #' ## [1,]    2    3    4    5
  #' ## [2,]    3    4    5    6
  varying_rows <- matrix_ %>%
    varying_row_mask()
  matrix_[varying_rows,]
}

ComBat_ignore_nonvariance <- function(matrix_, batch)
{
  #' Run ComBat and ignore nonvarying features.
  #'
  #' ComBat requires that all features have some variance (and probably assumes
  #' that all features are normally distributed). Since some features don't
  #' vary across samples, this function ignores nonvarying features before
  #' running ComBat.
  #'
  #' @param matrix_ The matrix to batch adjust with ComBat.
  #' @param batch The per-sample batch assignments. See the ComBat function for
  #' more information.
  #' 
  #' @return The matrix_ after batch adjustment.
  #'
  #' @examples
  #' ComBat_ignore_nonvariance(data, c(rep(1, 5000), rep(2, 5000)))
  variance_mask <- varying_row_mask(matrix_)
  varying_rows <- remove_nonvarying_rows(matrix_)
  adjusted <- ComBat(varying_rows, batch)
  matrix_[variance_mask,] <- adjusted
  matrix_
}

batch_adjust_tidy <- function(df, adjuster = ComBat_ignore_nonvariance, batch_col = "Batch") {
  categorical <- df %>%
    select_if(~!is.numeric(.) || is.whole(.))
  quantitative <- df %>%
    select_if(~is.numeric(.) && !is.whole(.))
  
  adjusted <- quantitative %>% t() %>% as.matrix() %>% adjuster(df[[batch_col]]) %>% t() %>% as.tibble()
  bind_cols(categorical, adjusted)
}