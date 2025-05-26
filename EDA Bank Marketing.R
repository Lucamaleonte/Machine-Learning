# Load libraries
library(tidyverse)
library(caret)
library(pROC)
library(GGally)
library(car)
library(glmnet)
library(smotefamily)
library(FNN)

# Set working directory and load data
setwd('C:/Users/lucaf/OneDrive/Desktop')
data <- read.csv("bank.csv", sep = ";", stringsAsFactors = FALSE)

# Basic information
head(data)
str(data)
summary(data)

# Preprocessing
categorical_vars <- c("job", "marital", "education", "default", "housing",
                      "loan", "contact", "month", "day", "poutcome", "y")
data[categorical_vars] <- lapply(data [ categorical_vars ], as.factor)
data$previous_contact <- factor(ifelse(data$pdays == 999, "no", "yes"))
data <- data %>% select(-duration)

str(data)
summary(data)
table(data$previous_contact)
colSums(is.na(data))

# Exploratory Data Analysis and Statistical Tests

## Age Distribution
ggplot(data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  theme_minimal() +
  labs(title = "Age Distribution", x = "Age", y = "Count")

## Subscription Count
ggplot(data, aes(x = y)) +
  geom_bar(fill = "lightgreen", color = "black") +
  theme_minimal() +
  labs(title = "Subscription Count", x = "Subscription (y)", y = "Count")

yes_count <- sum(data$y == "yes")
no_count  <- sum(data$y == "no")
print(prop.test(x = yes_count, n = yes_count + no_count, p = 0.5))

## Age Distribution by Subscription Status
ggplot(data, aes(x = y, y = age)) +
  geom_boxplot(fill = "orange", color = "black") +
  theme_minimal() +
  labs(title = "Age by Subscription", x = "Subscription (y)", y = "Age")


print(t.test(age ~ y, data = data))
print(wilcox.test(age ~ y, data = data))

## Job Category Distribution by Subscription
ggplot(data, aes(x = job, fill = y)) +
  geom_bar(position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Job Category by Subscription", x = "Job", y = "Count")

job_table <- table(data$job, data$y)
print(job_table)
print(chisq.test(job_table))

## Distribution of Campaign Contacts
ggplot(data, aes(x = campaign)) +
  geom_histogram(binwidth = 1, fill = "purple", color = "black") +
  theme_minimal() +
  labs(title = "Campaign Contacts", x = "Contacts", y = "Count") +
  scale_x_continuous(limits = c(0, 20)) 


print(summary(data$campaign[data$y == "yes"]))
print(summary(data$campaign[data$y == "no"]))
print(wilcox.test(campaign ~ y, data = data))

##Subscription Rate by Campaign Contacts
campaign_rates <- data %>%
  group_by(campaign) %>%
  summarise(
    total = n(),
    yes_count = sum(y == "yes"),
    subscription_rate = yes_count / total
  ) %>%
  arrange(campaign)
print(head(campaign_rates, 10))

ggplot(campaign_rates, aes(x = campaign, y = subscription_rate)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Subscription Rate by Contacts", x = "Contacts", y = "Rate") +
  scale_x_continuous(limits = c(0, 40))   


##Marital Status Distribution and Subscription Rates
ggplot(data, aes(x = marital, fill = y)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(
    title = "Proportion of Subscription by Marital Status",
    x = "Marital Status",
    y = "Proportion"
  )
marital_table <- table(data$marital, data$y)
print(marital_table)
print(chisq.test(marital_table))

##Education Level Distribution and Subscription Rates
ggplot(data, aes(x = education, fill = y)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(
    title = "Proportion of Subscription by Education Level",
    x = "Education",
    y = "Proportion"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Manual counts and proportion plots for contact, month, day_of_week, poutcome
contact_df <- data.frame(
  contact = c("cellular", "telephone"),
  no = c(22291, 14257),
  yes = c(3853, 787)
)
contact_long <- contact_df %>%
  pivot_longer(cols = c("no", "yes"), names_to = "subscription", values_to = "count")
ggplot(contact_long, aes(x = contact, y = count, fill = subscription)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Subscription Counts by Contact Type", x = "Contact Type", y = "Count")

month_df <- data.frame(
  month = c("apr", "aug", "dec", "jul", "mar", "may"),
  no = c(2093, 5523, 93, 6525, 270, 12883),
  yes = c(539, 655, 89, 649, 276, 886)
)
month_long <- month_df %>%
  pivot_longer(cols = c("no", "yes"), names_to = "subscription", values_to = "count")
ggplot(month_long, aes(x = month, y = count, fill = subscription)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Subscription Counts by Month", x = "Month", y = "Count")

contact_df <- contact_df %>%
  mutate(
    Total   = no + yes,
    no_prop = no / Total,
    yes_prop = yes / Total
  )
contact_prop_long <- contact_df %>%
  select(contact, no_prop, yes_prop) %>%
  pivot_longer(cols = c("no_prop", "yes_prop"), names_to = "subscription", values_to = "proportion")
ggplot(contact_prop_long, aes(x = contact, y = proportion, fill = subscription)) +
  geom_bar(stat = "identity", position = "fill") +
  theme_minimal() +
  labs(title = "Subscription Proportions by Contact Type", x = "Contact Type", y = "Proportion")



poutcome_df <- data.frame(
  poutcome = c("failure", "nonexistent", "success"),
  no = c(3647, 32422, 479),
  yes = c(605, 3141, 894)
)
poutcome_long <- poutcome_df %>%
  pivot_longer(cols = c("no", "yes"), names_to = "subscription", values_to = "count")
ggplot(poutcome_long, aes(x = poutcome, y = count, fill = subscription)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(
    title = "Subscription Counts by Outcome of Previous Campaign ",
    x = "Previous Outcome",
    y = "Count"
  )
