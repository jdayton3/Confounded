# Get command line args ---------------------------

# args = commandArgs(trailingOnly = TRUE)
# 
# if (length(args) != 1) {
#   stop("Must supply exactly one argument.\nUsage:\n$ Rscript combat.R path/to/file.csv\n")
# }
# input_path = args[1]
input_path = "../../data/mnist/noisy.csv"

# Load our stuff -----------------------------------

source("functions.R")
suppressMessages(library(stringr))

# Get the output path ---------------------------

# /path/to/thing.csv -> /path/to/thing_sva.csv
output_path <- str_replace_all(input_path, "(^.*)(\\.csv$)", "\\1_sva\\2")

# Run ComBat ------------------------------------

df <- read_csv(input_path) %>% head(100)
adj <- batch_adjust_tidy(df, batch_col="Batch", adjuster=sva_ignore_nonvariance) #%>% write_csv(output_path)
