library(tools)
library(keras)
library(quanteda)
source(file_path_as_absolute("utils.R"))
source(file_path_as_absolute("resultsHelper.R"))

for (i in 1:2) {

  source(file_path_as_absolute("getDados.R"))

  ### Rede
  
  embedding_dims <- 100
  filters <- 132
  
  main_input <- layer_input(shape = c(maxlen), dtype = "int32")
  
  embedding_input <- 	main_input %>% 
    layer_embedding(input_dim = vocab_size, output_dim = embedding_dims, input_length = maxlen, name = "embedding")
  
  ccn_out_3 <- embedding_input %>% 
    layer_conv_1d(
      filters, 3,
      padding = "valid", activation = "relu", strides = 1
    ) %>%
    layer_global_max_pooling_1d()
  
  ccn_out_4 <- embedding_input %>% 
    layer_conv_1d(
      filters, 4, 
      padding = "valid", activation = "relu", strides = 1
    ) %>%
    layer_global_max_pooling_1d()
  
  ccn_out_5 <- embedding_input %>% 
    layer_conv_1d(
      filters, 5, 
      padding = "valid", activation = "relu", strides = 1
    ) %>%
    layer_global_max_pooling_1d()
  
  main_output <- layer_concatenate(c(ccn_out_3, ccn_out_4, ccn_out_5)) %>% 
    layer_dropout(0.1) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 6, activation = 'softmax')
  
  model <- keras_model(
    inputs = c(main_input),
    outputs = main_output
  )
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  history <- model %>% fit(
    dados_train,
    to_categorical(dataTrain$categoria),
    epochs = 10,
    batch_size = 64,
    validation_split = 0.15
    #,
    #callback = list(
    #  callback_early_stopping(
    #    monitor = "val_loss",
    #    patience = 5
    #  )
    #)
  )
  
  history
  
  predictions <- model %>% predict(dados_test)
  predictionsMax <- apply(predictions, 1, which.max) - 1
  matriz <- confusionMatrix(as.factor(dataTest$categoria), as.factor(predictionsMax))
  addResult(matriz)
  resultados
}

dumpResults()