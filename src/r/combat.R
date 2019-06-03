#Debugging: 
options(error=function() { traceback(2); if(!interactive()) quit("no", status = 1, runLast = FALSE) })

# Load dependencies --------------------------------

load_stuff <- function() {
  for (package in c("tidyverse", "docstring", "stringr", "argparse")) {
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

# Parse command line args --------------------------

parser <- ArgumentParser()

parser$add_argument("input_file", help = "path to the input file")
parser$add_argument("-a", "--adjuster", default = "combat", choices = c("combat", "scale"), help = "method to use for adjustment")
parser$add_argument("-b", "--batch-col", default = "Batch", help = "title of batch column to adjust for")
parser$add_argument("-o", "--output-file", help = "path to the output file. default: [input_file]_combat.csv")

args <- parser$parse_args()

if (is.null(args$output_file)) {
  # /path/to/thing.csv -> /path/to/thing_combat.csv
  args$output_file <- output_path <- str_replace_all(args$input_file, "(^.*)(\\.csv$)", sprintf("\\1_%s\\2", args$adjuster))
}


# Define functions ---------------------------------

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
    as_tibble() %>%
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
  #' @param matrix_ The matrix to batch adjust with ComBat. Columns are features,
  #' rows are samples.
  #' @param batch The per-sample batch assignments. See the ComBat function for
  #' more information.
  #' 
  #' @return The matrix_ after batch adjustment.
  #'
  #' @examples
  #' ComBat_ignore_nonvariance(data, c(rep(1, 5000), rep(2, 5000)))
  matrix_ <- t(matrix_)
  variance_mask <- varying_row_mask(matrix_)
  varying_rows <- remove_nonvarying_rows(matrix_)
  adjusted <- ComBat(varying_rows, batch)
  matrix_[variance_mask,] <- adjusted
  t(matrix_)
}

scale_adjust <- function(matrix_, batch)
{
  #' Run ComBat and ignore nonvarying features.
  #'
  #' @param matrix_ The matrix to batch adjust by scaling. Columns are
  #' features, rows are samples.
  #' @param batch The per-sample batch assignments.
  #' 
  #' @return The matrix_ after batch adjustment.
  #'
  #' @examples
  #' scale_adjust(data, c(rep(1, 5000), rep(2, 5000)))
  
  # Get columnwise mins & maxes
  mins <- apply(matrix_, 2, min)
  maxes <- apply(matrix_, 2, max)
  # Scale each batch individually to [0, 1]
  batches <- list()
  for (b in levels(factor(batch))) {
    # drop=F makes it return a matrix when you only grab one row.
    batch_rows <- matrix_[batch == b, , drop = FALSE]
    if (nrow(batch_rows) == 1) {
      stop(sprintf("Can't scale columns: batch '%s' only has 1 sample.", b))
    }
    batch_mins <- apply(batch_rows, 2, min)
    batch_maxes <- apply(batch_rows, 2, max)
    adjusted <- t((t(batch_rows) - batch_mins) / (batch_maxes - batch_mins))
    batches[[b]] <- adjusted
    # Merge adjustment back in
    matrix_[batch == b] = adjusted
  }
  # Scale back up to [min, max]
  t(t(matrix_) * (maxes - mins) + mins)
}

batch_adjust_tidy <- function(df, adjuster = ComBat_ignore_nonvariance, batch_col = "Batch") {
  categorical <- df %>%
    select_if(~!is.numeric(.) || is.whole(.))
  quantitative <- df %>%
    select_if(~is.numeric(.) && !is.whole(.))
  
  adjusted <- quantitative %>% as.matrix() %>% adjuster(df[[batch_col]]) %>% as_tibble()
  bind_cols(categorical, adjusted)
}

# Run the adjuster ------------------------------------

adjusters <- list(
  combat = ComBat_ignore_nonvariance,
  scale = scale_adjust
)
adjuster <- adjusters[[args$adjuster]]

message("Reading input file.")

suppressMessages(df <- read_csv(args$input_file))

if (!(args$batch_col %in% names(df))) {
  discrete_col_names <- df %>%
    select_if(~!is.numeric(.) || is.whole(.)) %>%
    names()
  error_message <- sprintf(
    "--batch-col argument (default 'Batch', selected '%s') must be a column in 'input_path' csv. Options: [%s]",
    args$batch_col,
    paste(discrete_col_names, collapse = ", ")
  )
  stop(error_message)
}

message(sprintf("Adjusting using the '%s' adjuster", args$adjuster))
print(df)
batch_adjust_tidy(
  df, 
  batch_col = args$batch_col, 
  adjuster = adjuster
) %>% write_csv(args$output_file)

message(sprintf("Saved output to '%s'", args$output_file))