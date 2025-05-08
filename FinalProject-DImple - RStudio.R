# Load necessary libraries
library(dplyr)
library(nnet)
library(ggplot2)
install.packages("stargazer")
library(stargazer)

# Load the dataset
AB_data_2019 <- read.csv("C:/Users/reshm/OneDrive - University of Massachusetts Boston/3rd Semester/MultiVariant and Regression/Final Project/Dataset/AB_NYC_2019.csv")
colnames(AB_data_2019)

# Categorizing price into three categories: Low, Medium, High
AB_data_2019$price_category <- cut(AB_data_2019$price, 
                                   breaks = c(0, 50, 150, Inf), 
                                   labels = c("Low", "Medium", "High"))

# Remove outliers from price using the 1.5 * IQR rule
Q1 <- quantile(AB_data_2019$price, 0.25)
Q3 <- quantile(AB_data_2019$price, 0.75)
IQR <- Q3 - Q1
AB_data_2019 <- AB_data_2019 %>%
  filter(price >= (Q1 - 1.5 * IQR) & price <= (Q3 + 1.5 * IQR))

# Convert categorical variables to factors
AB_data_2019$neighbourhood_group <- as.factor(AB_data_2019$neighbourhood_group)
AB_data_2019$room_type <- as.factor(AB_data_2019$room_type)

# Split the data into training and test sets (80% train, 20% test)
set.seed(123)
trainIndex <- sample(1:nrow(AB_data_2019), 0.8 * nrow(AB_data_2019))
trainData <- AB_data_2019[trainIndex, ]
testData <- AB_data_2019[-trainIndex, ]

# Multinomial logistic regression model
model <- multinom(price_category ~ neighbourhood_group + neighbourhood + latitude + longitude + 
                    room_type + minimum_nights + number_of_reviews + reviews_per_month + 
                    calculated_host_listings_count + availability_365, 
                  data = trainData)

# Summarizing the model to get p-values
summary_model <- summary(model)

coefficients <- summary_model$coefficients
z_values <- summary_model$coefficients / summary_model$standard.errors  # Compute z-values

# Calculate p-values from z-values (two-tailed test)
p_values <- 2 * (1 - pnorm(abs(z_values)))

# Print p-values
p_values
# Summarize the model using the Stargazer package for a tabular format
stargazer(model, type = "text", title = "Multinomial Logistic Regression Results", out = "model_results.txt")

# Predictions on test data
predictions <- predict(model, newdata = testData)

# Check the first few predictions
head(predictions)

# Visualizations

# Filter data to remove extreme outliers for reviews_per_month
filtered_data <- AB_data_2019 %>%
  filter(reviews_per_month > 0 & reviews_per_month < 15)

# Histogram + Density plot for reviews_per_month
ggplot(filtered_data, aes(x = reviews_per_month)) +
  geom_histogram(aes(y = ..density..), binwidth = 0.5, fill = "lightblue", color = "black", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  theme_minimal() +
  labs(title = "Reviews per Month Distribution", x = "Reviews per Month", y = "Density")

# Boxplot for price by room type
ggplot(AB_data_2019, aes(x = room_type, y = price, fill = room_type)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Price by Room Type", x = "Room Type", y = "Price")


# Subset data by room type
entire_home_apt <- subset(AB_data_2019, room_type == "Entire home/apt")
private_room <- subset(AB_data_2019, room_type == "Private room")
shared_room <- subset(AB_data_2019, room_type == "Shared room")


# Create a combined data frame to draw multiple histograms
entire_ha <- data.frame(price = entire_home_apt$price, group = "Entire home/apt")
private_r <- data.frame(price = private_room$price, group = "Private room")
shared_r <- data.frame(price = shared_room$price, group = "Shared room")

combined_data <- rbind(entire_ha, private_r, shared_r)

# Define custom bins for price
bin_breaks <- seq(0, 500, by = 10)

# Plot the histogram
ggplot(combined_data, aes(x = price, fill = group)) +
  geom_histogram(breaks = bin_breaks, alpha = 0.6, position = "identity", color = "black") +
  scale_fill_manual(
    values = c("Entire home/apt" = "red", "Private room" = "green", "Shared room" = "blue")
  ) +
  labs(
    title = "Price Distribution by Room Type",
    x = "Price",
    y = "Frequency",
    fill = "Room Type" # Legend title
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right" # Ensures the legend is on the side
  )

# Histogram for price
ggplot(AB_data_2019, aes(x = price)) +
  geom_histogram(binwidth = 50, fill = "blue", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Histogram of Price", x = "Price", y = "Frequency")

# Violin plot: Price distribution by room type
ggplot(AB_data_2019, aes(x = room_type, y = price, fill = room_type)) +
  geom_violin(trim = FALSE) +
  labs(title = "Price Distribution by Room Type", x = "Room Type", y = "Price") +
  theme_minimal()
