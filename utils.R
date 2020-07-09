library(magrittr)
library(tm)

csvRead <- function(dir, rows_to_read){
  data = data.table::fread(dir, nrows=rows_to_read)
  return (data)
}