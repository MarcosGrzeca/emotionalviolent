library(caret)

getHeader <- function() {
  return (c("P-Surprise", "P-Sadness", "P-None", "P-Fear", "P-Disgust", "P-Anger", "R-Surprise", "R-Sadness", "R-None", "R-Fear", "R-Disgust", "R-Anger", "F1-Surprise", "F1-Sadness", "F1-None", "F1-Fear", "F1-Disgust", "F1-Anger"));
}

resultados <- data.frame(matrix(ncol = 18, nrow = 0))
names(resultados) <- getHeader()

addResult <- function(matriz) {
	precision <- as.data.frame(matriz$byClass[, 'Pos Pred Value'])
	recall <- as.data.frame(matriz$byClass[, 'Sensitivity'])
	result <- c()
	for (i in 0:6) {
	  result <- c(result, precision[i,1] * 100)
	}

	for (i in 0:6) {
	  result <- c(result, recall[i,1] * 100)
	}

	for (i in 0:6) {
	  result <- c(result, 100 * ( 2 * precision[i,1] * recall[i,1] / (precision[i,1] + recall[i,1])))
	}

	names(result) <- getHeader()
	resultados <<- rbind(resultados, result)
	names(resultados) <<- getHeader()

}

dumpResults <- function(fileName) {
	write.table(resultados, file = fileName, sep = "\t",
            row.names = TRUE, col.names = NA, quote = FALSE, append = TRUE)
}