library(readxl)
library(stringr)
library(nltk)
library(tm)
library(SnowballC)
library(caret)
library(e1071)
library(class)


# loading dataset
setwd("C:/Users/nandu/Desktop/WMU/spring1/ML/v")
data <- read_excel("SMSSpamCollection.xlsx")
msg <- data.frame(type = data[,1], message = data[,2])
colnames(msg) <- c("type", "message")
msg$type <- as.factor(msg$type)
msg$token <- as.numeric(msg$type == "spam")
# Preprocessing the data
corpus <- VCorpus(VectorSource(msg$message))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)

# creating document term matrix
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.995)
msg_corpus <- Corpus(VectorSource(msg$message))
msg_corpus_clean <- tm_map(msg_corpus, removePunctuation)
msg_corpus_clean <- tm_map(msg_corpus_clean, content_transformer(tolower))
msg_corpus_clean <- tm_map(msg_corpus_clean, removeNumbers)
msg_corpus_clean <- tm_map(msg_corpus_clean, removeWords, stopwords("english"))
msg_corpus_clean <- tm_map(msg_corpus_clean, stemDocument)
msg_dtm <- DocumentTermMatrix(msg_corpus_clean)
msg_matrix_stem <- as.matrix(msg_dtm)


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
model2 <- glm(type_trained_stem ~ ., data = as.data.frame(msg_trained_stem), family = binomial)
# Make predictions on test data
pred2 <- predict(model2, newdata = as.data.frame(msg_test_stem), type = "response")

# Convert predicted probabilities to binary predictions
pred2_binary <- ifelse(pred2 > 0.5, 1, 0)
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
cat("Model accuracy using KNN with k = 5: ", model3_accuracy, "\n")
cat("Model accuracy using KNN with k = 15: ", model4_accuracy, "\n")
