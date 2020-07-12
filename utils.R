library(magrittr)
library(tm)

get_os <- function(){
  sysinf <- Sys.info()
  if (!is.null(sysinf)){
  os <- sysinf['sysname']
  if (os == 'Darwin')
    os <- "osx"
  } else { ## mystery machine
    os <- .Platform$OS.type
    if (grepl("^darwin", R.version$os))
      os <- "osx"
    if (grepl("linux-gnu", R.version$os))
      os <- "linux"
  }
  tolower(os)
}

csvRead <- function(dir, rows_to_read){
  data = data.table::fread(dir, nrows=rows_to_read)
  return (data)
}