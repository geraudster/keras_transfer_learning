#' Inspired from https://keras.rstudio.com/
#'
#'

#' ## Installation
#'
#+ keras_install, cache=TRUE
devtools::install_github("rstudio/keras")


#' Keras uses Tensorflow by default. Following lines will install
#' Keras library with Tensorflow backend:
library(keras)
use_python("/usr/bin/python3")

#+ keras_tf_install, cache=TRUE
install_keras()

library(reticulate)
py_install('pydot', envname = 'r-tensorflow')

#' You might encounter the following error
#' > ImportError: No module named _internal
#' 
#' This can be resolved by upgradin pip:
#' `sudo pip install --upgrade pip`

#' ## MNIST problem
#' MNIST is a dataset of hand-written digits. The objective is to train a
#' model able to recognize those digits

#' We'll start by loading the dataset and display some images:

mnist <- dataset_mnist()

par(mfrow=c(2,5)) 
imgs <- sapply(1:10, function (i) {
    image(t(apply(mnist$train$x[i,,], 2, rev)))
})

#' ## Data preparation
#' As usual we split our dataset in train/test set:

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

#' Images have size 28x28, so we will first flatten the image to have r`28x28` length vector

## reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

#' Next we need to *one-hot encode* the labels:
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

#' We're now ready to train a model !
#'
#' ## Model definition
#'
#' We start by reproducing results from https://keras.rstudio.com/:

model <- keras_model_sequential() 
model %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')

#' ```{python}
#' from keras.utils import plot_model
#' plot_model(r.model, to_file='model.png')
#' ```

knitr::include_graphics('./model.png')

#' Next we compile the model:
model %>% compile(
              loss = 'categorical_crossentropy',
              optimizer = optimizer_rmsprop(),
              metrics = c('accuracy')
          )


#' ## Model training
#' We can now train our model. Doing this with Keras is quite easy, we can call the `fit` function.
#+ first_model, cache=TRUE
history <- model %>% fit(
                         x_train, y_train, 
                         epochs = 30, batch_size = 128, 
                         validation_split = 0.2
                     )
#' After a few minutes, our model is trained.
#' `fit` method returns statistics about training, we can plot it:
plot(history)

#' Keras also provides functions for model performance evaluation:
model %>% evaluate(x_test, y_test)

#' ![Scratching head](https://media.giphy.com/media/VUdLlDsKlQi5i/giphy.gif)
#' 
#' Something's wrong... accuracy on [Keras first steps](https://keras.rstudio.com/#training-and-evaluation)
#' is `0.9807` and we only have an accuracy near `0.85`.
#' We forgot the normalization step, so let's retrain a model with normalized data and see what happen:
#+ second_model, cache=TRUE
x_train <- x_train / 255
x_test <- x_test / 255

history <- model %>% fit(
                         x_train, y_train, 
                         epochs = 30, batch_size = 128, 
                         validation_split = 0.2
                     )
plot(history)
model %>% evaluate(x_test, y_test)

#' It looks better, now let's try some models with different architecture.
#' But first let's create a function that will help us evaluate an architecture:

evaluate_architecture <- function(model) {
    history <- model %>% compile(
                             loss = 'categorical_crossentropy',
                             optimizer = optimizer_rmsprop(),
                             metrics = c('accuracy')) %>%
        fit(x_train, y_train, 
            epochs = 30, batch_size = 128, 
            validation_split = 0.2)
    list(training_plot = plot(history), score = model %>% evaluate(x_test, y_test))
}

#'
#' ## Without dropout
#' Dropout layers helps in reducing overfitting by randomly removing
#' connections between layers at training time. Let's see what happen
#' if we remove the dropout layers.
#+ model_without_dropout, cache=TRUE

model_without_dropout <- keras_model_sequential() %>% 
    layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')

evaluate_architecture(model_without_dropout)

#' TODO
#'

#+ model_with_128_cells, cache=TRUE
model_with_128_cells <- keras_model_sequential()  %>% 
    layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% 
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')

evaluate_architecture(model_with_128_cells)

#' ## Size matters ?

model_with_128_cells_1_layer <- keras_model_sequential() %>% 
    layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% 
    layer_dense(units = 10, activation = 'softmax')

evaluate_architecture(model_with_128_cells_1_layer)

#' ## Try different cells number
#+ models_nb_cells, cache=TRUE
results_nb_cells <- lapply(4:10, function (i) {
    model <- keras_model_sequential() %>% 
        layer_dense(units = 2^i, activation = 'relu', input_shape = c(784)) %>% 
        layer_dense(units = 10, activation = 'softmax')
    
    evaluate_architecture(model)[['score']]
    })

#' The results:
results_nb_cells

library(ggplot2)
as.dataframe(results_nb_cells)
