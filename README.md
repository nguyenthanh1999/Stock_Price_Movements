# **Stock Price Movements**

## **Introduction**
### **Purpose of document implementation**
My aim is to predict stock market fluctuations based on the news during that time period. Because stocks are affected by many objective factors and news is only a part of them, it is impossible to predict too much based on news alone. So within the scope of this project, I only predict whether news can change stock values or not.

Although it is currently only possible to expect how news in a specific period of time will affect stock value, it will also be a valuable reference source for stock buyers to preliminary assess market volatility prices based on recent news sources. 

Later, with models with more complex algorithms and more complete information, the problem can be expanded towards more specific predictions.

### **The goal of the subject**
Build a model to predict whether news about a company will affect the value of that company's stock. The dataset will consist of a summary of news scraped from Google to create features. Labels will be assigned based on the fluctuation of the stock value, with increases and decreases marked as 'changed'.

## **Methodology**
For collecting data, I used Selenium, the urllib.parse module, the quote() function, and other techniques to scrape data from websites. I then summarized and converted it to create the entire dataset.

As for the model, I utilized knowledge gathered from research to engineer features, making it easier for the algorithm to process. I employed models from the sklearn library, such as Logistic Regression, SVM, Random Forest, and k-Nearest Neighbors, for testing. Finally, I used GridSearchCV to fine-tune the best model and improve performance.

## **Dataset**
### **Data Collection**
To collect data for creating a dataset, I utilize the Selenium library and its packages to develop a function capable of scraping data from Google's news page over a specific period of time. This allows me to retrieve information about prominent articles related to the search goal, including title, article link, snippet, publication time, etc. Additionally, I summarize the content of each article by leveraging other libraries.

*Here is the link to dataset creator:* [dataset_creator.ipynb](dataset_creator.ipynb)

There's few thing to be concern when you using my dataset creator:
-   You need to download the webdriver that is suitable for your browser and specify its path in the "webdriver_path" variable.
```python
webdriver_path = r'C:\WebDriver\chromedriver-win64\chromedriver.exe'
```
-   You need to download the [companyname_code.csv](companyname_code.csv) file to create VNM_stopwords for the extractive_summary function.
```python
def extractive_summary(text, vietnamese_stopwords,num_sentences=10):
    if len(text) > 200:
        sentences = sent_tokenize(text)
        words = nltk.word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in vietnamese_stopwords]
        freq_dist = FreqDist(filtered_words)
        sentence_scores = {}
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word in freq_dist.keys():
                    if len(sentence.split(' ')) < 30:
                        if sentence not in sentence_scores.keys():
                            sentence_scores[sentence] = freq_dist[word]
                        else:
                            sentence_scores[sentence] += freq_dist[word]
        summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        summary_sentences = sorted(summary_sentences, key=lambda x: sentences.index(x[0]))
        summary = ' '.join([sentence[0] for sentence in summary_sentences])
        return summary
    else:
        return text
```
-   Note that Google's HTML structure may change over time, so the CSS selectors used to find elements (such as "AZtVbe", "SoaBEf", "CBAQAA", "SoaBEf.R24aHf" for the articles class name, 'n0jPhd ynAwRc MBeuO nDgy9d' for the title class name, 'GI74Re nDgy9d' for the snippet class name, 'OSrXXb rbYSKb LfVVr' for the publication date class name) may become outdated. Regular maintenance may be required to update the selectors accordingly.

*Ensure to check for CAPTCHA pages and prompt intervention if a CAPTCHA is detected, as manual solving may be necessary.*

Here's the dataset when you finish using dataset_creator:

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\10.Dataset.png" alt="Dataset">
</div>

## **Data Preprocessing**
After some checking process for data information, I renamed the 'Fluctuations' column to label the output with two types of data: 'CHANGED' and 'UNCHANGED'. I then visualized it using a countplot to check its balance.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\14.Fluctuations_visualization.png" alt="Label countplot">
</div>

With the 'True' label representing the 'CHANGED' price, meaning 'INCREASE' or 'DECREASE', and the 'FALSE' label representing the 'UNCHANGED' price, it seems quite balanced.

This is the wordcloud feature of "CHANGED" and "UNCHANGED" labels

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\15.Wordclouds.png" alt="Wordclouds">
</div>

I need to clean this data to create better features for my models. I cleaned it by lowercasing all the text, removing square bracket content, URLs, HTML tags, punctuation, newlines, alphanumeric words, and stopwords. Then, I joined it back into a single string and assigned it to the new columns I have created. And here is the results

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\18.Cleaning_result.png" alt="Data Cleaned">
</div>

## **Feature Engineering**
I created two lists of positive and negative words for the purpose of classifying features. Additionally, I also created two more lists of antonyms of the original two lists by adding negative prefixes before them, such as 'không', 'chưa', 'chẳng', 'chả', and so on. Along with that, I developed a few functions to support feature engineering.

Using functions to support feature engineering that I've built before, I applied them to each feature of the cleaned data. I created features of positive and negative points for each cleaned feature. Then, I summed each pair of these points of the same cleaned feature to create the positivity feature for each news feature from the base dataframe.

Next, I use the 'positivity' columns to create the features (X) and the 'Fluctuations' column to create the output (y) of the model. Then, I use the train_test_split() function to split it into a train set and a validation set.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\20.positivity_each_feature.png" alt="Feature Engineered">
</div>

Now that I have my cleaned X and y for training and validation sets, I used them to train my models.

## **Results**
As mentioned above, within the scope of this project, I will use Logistic Regression, SVM, Random Forest, and k-Nearest Neighbors models for testing because it is a classification problem. Applying the features that I've engineered above, here are the results.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\21.Results_of_models.png" alt="Models results">
</div>

All results, although not high, are still acceptable. While the logistic regression model has the lowest accuracy results, the other three models show higher results that are quite similar. However, the SVM and Random Forest models appear to be overfit because the training set accuracy is much higher than the test set accuracy. Additionally, since the main objective is to correctly predict whether the stock price will increase or decrease (i.e., binary classification), rather than predicting the exact price, metrics like precision, recall, and F1-score become more important as they measure the model's ability to correctly classify price movements. The k-nearest neighbor model has the highest F1-score of all models. Therefore, I have chosen the k-nearest neighbor model for this project.

### **Fine-tuning**
After choosing the k-nearest neighbor model for this project, I attempted to improve predictions by using the GridSearchCV package to tune hyperparameters such as 'n_neighbors', 'weights', 'algorithm', 'leaf_size', 'metric' and 'p' to find the best model based on new combinations of these hyperparameters.

GridSearchCV in scikit-learn is a method used for hyperparameter tuning through an exhaustive search over a specified parameter grid. It helps find the best set of hyperparameters for a model.

After fitting the GridSearchCV object to your data, you can access the best parameters and best estimator using 'best_params_' and 'best_estimator_', respectively. Use the best model for predictions or further analysis. Adjust the parameters and dataset according to your specific use case.

Set up hyperparameters and folds for cross-validation. The GridSearchCV package will help find the best combination of these hyperparameters to identify the best model.

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\22.GridSearchCV_setup.png" alt="Fine-tuning">
</div>

Here are the results of the best model found by GridSearchCV based on the combination of hyperparameters:

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\23.Results_after_tune.png" alt="GridSearchCV results">
</div>

<div style="text-align:center; margin-bottom:20px;">
    <img src="Images\24.Final_ROC_curve.png" alt="ROC curve">
</div>

The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. It evaluates the performance of a classifier across different threshold values for binary classification problems.

## **Conclusion**
Overall, the results, although not high, are still acceptable. The relatively lower performance may be attributed to the small size of the dataset, indicating a need for more data to better train the model. However, the most significant factor could be that news is not the sole determinant of stock value changes; there are numerous other influential factors. Therefore, the current model's complexity might not be sufficient to accurately predict these changes.

*Read my report for more details* [report.docx](report.docx)
