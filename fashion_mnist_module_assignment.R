# Import necessary libraries
library(keras)
library(tensorflow)
library(ggplot2)
library(reshape2)

# Load the Fashion MNIST dataset
mnist <- dataset_fashion_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Preprocess the data
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1)) / 255
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1)) / 255

# Preview one of the images
img <- matrix(train_images[1, , , ], nrow = 28, ncol = 28)
ggplot(melt(img), aes(Var1, Var2, fill = value)) + 
  geom_tile() + 
  scale_fill_gradient(low = "white", high = "black") + 
  ggtitle(paste("Label:", train_labels[1])) +
  theme_minimal()

# Define the CNN model
model <- keras_model_sequential() %>%
  # Layer 1: Convolutional layer
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  # Layer 2: MaxPooling layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # Layer 3: Convolutional layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  # Layer 4: MaxPooling layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # Layer 5: Convolutional layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  # Layer 6: Flatten layer
  layer_flatten() %>%
  # Dense layers for classification
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
model %>% fit(train_images, train_labels, epochs = 5, batch_size = 64)

# Evaluate the model
score <- model %>% evaluate(test_images, test_labels)
cat('Test accuracy:', score['accuracy'], '\n')

# Make predictions for two images
predictions <- model %>% predict(test_images[1:2, , , ])
cat('Predictions for first two images:\n')
print(predictions)

# Visualize the predictions
for (i in 1:2) {
  img <- matrix(test_images[i, , , ], nrow = 28, ncol = 28)
  p <- ggplot(melt(img), aes(Var1, Var2, fill = value)) + 
    geom_tile() + 
    scale_fill_gradient(low = "white", high = "black") + 
    ggtitle(paste("Predicted label:", which.max(predictions[i, ]))) +
    theme_minimal()
  print(p)  # Explicitly print the plot
}
