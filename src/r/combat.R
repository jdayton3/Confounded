# Get command line args ---------------------------

args = commandArgs(trailingOnly = TRUE)

args = c("../../data/avery/GSE39582/tidy.csv")

if (length(args) != 1) {
  stop("Must supply exactly one argument.\nUsage:\n$ Rscript combat.R path/to/file.csv\n")
}
input_path = args[1]

# Load our stuff -----------------------------------

load_stuff <- function() {
  if (!require("pacman")) install.packages("pacman")
  p_load("tidyverse", "docstring", "stringr")

  install_sva <- function() {
    ## try http:// if https:// URLs are not supported
    source("https://bioconductor.org/biocLite.R")
    biocLite("sva")
  }
  if (!require("sva")) install_sva()
  library(sva)
  source("functions.R")
}
suppressMessages(load_stuff())


# Get the output path ---------------------------

# /path/to/thing.csv -> /path/to/thing_combat.csv
output_path <- str_replace_all(input_path, "(^.*)(\\.csv$)", "\\1_combat\\2")

# Run ComBat ------------------------------------

df <- read_csv(input_path)
batch <- df$Batch

## Find the categorical columns & separate them

categorical <- df %>%
  select_if(~!is.numeric(.) || is.integer(.))
quantitative <- df %>%
  select_if(~is.numeric(.) && !is.integer(.))

## Run ComBat on the quantitative columns
adjusted <- quantitative %>% t() %>% as.matrix() %>% ComBat_ignore_nonvariance(batch) %>% t() %>% as.tibble()

## Stick them back together & save them
bind_cols(categorical, adjusted) %>% write_csv(output_path)
