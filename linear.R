library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(tm)
library(e1071)
library(caret)

# Loading dataset
data <- read_excel("SMSSpamCollection.xlsx", col_names = FALSE)
msg <- data %>% 
  rename(type = ...1, message = ...2)
msg$token <- as.numeric(msg$type == "spam")

# Preprocessing the data
preprocessing <- function(text){
  text <- gsub("[[:punct:]]", " ", text)
  text <- tolower(text)
  text <- removeWords(text, stopwords("english"))
  text <- removeNumbers(text)
  text <- stripWhitespace(text)
  return(text)
}

msg_copy_preprocessed <- msg$message %>%
  lapply(preprocessing) %>%
  unlist() %>%
  as.data.frame(stringsAsFactors = FALSE) %>%
  setNames("message")


# Applying stemming to the Data
stemming <- function(text){
  words <- strsplit(text, " ")[[1]]
  stemmed_words <- tm::stemDocument(words, language = "english")
  return(paste(stemmed_words, collapse = " "))
}

msg_copy_stem <- msg_copy_preprocessed$message %>% 
  lapply(stemming) %>% 
  as.data.frame() %>% 
  setNames("message")



library(tidytext)
library(tibble)

msg_tidy <- msg_copy_preprocessed %>%
  as.data.frame() %>%
  add_column(id = seq_len(nrow(.))) %>%
  unnest_tokens(word, message)


# Create a document-term matrix
dtm <- msg_tidy %>%
  count(id, word) %>%
  cast_dtm(document = id, term = word, value = n)


library(tidytext)
library(tidytext)
library(dplyr)
msg_tidy <- msg_copy_preprocessed %>%
  mutate(id = row_number()) %>%
  unnest_tokens(word, message)

msg_matrix_stem <- msg_tidy %>%
  count(id, word) %>%
  cast_sparse(id, word, n)

msg_matrix_stem <- as.matrix(dtm)

set.seed(20)
train_index <- createDataPartition(y = msg$token, p = 0.5, list = FALSE)
msg_trained_stem <- msg_matrix_stem[train_index, ]
msg_test_stem <- msg_matrix_stem[-train_index, ]
type_trained_stem <- msg$token[train_index]
type_test_stem <- msg$token[-train_index]

# Naive Bayes
model1 <- naiveBayes(x = msg_trained_stem, y = type_trained_stem, laplace = 1)
pred1 <- predict(model1, newdata = msg_test_stem)

# Accuracy of model using Naive Bayes
model1_accuracy <- mean(pred1 == type_test_stem)

# Logistic regression
model2 <- glm(type_trained_stem ~ ., data = msg_trained_stem, family = binomial)
pred2 <- predict(model2, newdata = msg_test_stem, type = "response") > 0.5

# Accuracy of model using Logistic Regression
model2_accuracy <- mean(pred2 == type_test_stem)

# KNN with k = 5
model3 <- knn(train = msg_trained_stem, test = msg_test_stem, cl = type_trained_stem, k = 5)
pred3 <- as.numeric(model3)

# Accuracy of KNN with k = 5
model3_accuracy <- mean(pred3 == type_test_stem)

# KNN with k = 15
model4 <- knn(train = msg_trained_stem, test = msg_test_stem, cl = type_trained_stem, k = 15)
pred4 <- as.numeric(model4)

# Accuracy of KNN with k = 15
model4_accuracy <- mean(pred4 == type_test_stem)

cat("Model accuracy using Naive Bayes: ", model1_accuracy, "\n")
cat("Model accuracy using Logistic regression: ", model2_accuracy, "\n")
cat("Model accuracy using KNN with k = 5: ", model3_accuracy, "\
