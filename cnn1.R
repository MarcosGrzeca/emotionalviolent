library(tools)
library(keras)
library(quanteda)
source(file_path_as_absolute("utils.R"))

dataTrain <- csvRead('datasets/MS_Treino.csv', 20000)
dataTest <- csvRead('datasets/MS_GS_v2.csv', 20000)

allTexts <- rbind(dataTrain, dataTest)

maxlen <- 70

tokenizer <-  text_tokenizer() %>%
  fit_text_tokenizer(allTexts$text)
vocab_size <- length(tokenizer$word_index) + 1
vocab_size

sequences_train <- texts_to_sequences(tokenizer, dataTrain$text)
dados_train <- pad_sequences(sequences_train, maxlen = maxlen)

sequences_test <- texts_to_sequences(tokenizer, dataTest$text)
dados_test <- pad_sequences(sequences_test, maxlen = maxlen)

### Rede

embedding_dims <- 128
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
  layer_dense(units = 64, activation = "relu") %>%
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
  epochs = 5,
  batch_size = 64,
  validation_split = 0.1
)

history

#results <- model %>% evaluate(dados_test, to_categorical(dataTest$categoria))
#results

predictions <- model %>% predict(dados_test)
predictions

require(caret)
#confusionMatrix(y_actual, y_predict)

a <- apply(predictions, 1, which.max)
aa <- a - 1

which.max(predictions[40,])

aa

matriz <- confusionMatrix(as.factor(dataTest$categoria), as.factor(aa))
matriz
