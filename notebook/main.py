
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import datetime
from transformers import AutoTokenizer # advanced tokenizer by hugging face
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax # smoothen the output between 0 and 1

# reading the roberta model
model_path = r"C:\Users\HP\Documents\NLP\model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def main():
    st.title('ReviewScope')
    col = st.columns(3)
    # importing the dataframe from the collection.py
    
    # function to classify the sentiment
    def remark(com):
        if com > 0.25:
            return 'positive'
        if com < -0.25:
            return 'negative'
        else:
            return 'neutral'
        
    # color review sentiment text
    def text_color(value):
        if value == 'negative':
            return "red" 
        elif value == 'positive':
            return "lightgreen"
        else:
            return "lightblue"
        
    # <--------------------- SIDE BAR---------------------->
    with st.sidebar:  
        col2 = st.columns(2) 
        lower = col2[0].selectbox('From:', [i for i in range(2004, 2024)])
        upper = col2[1].selectbox('To:', [i for i in range(lower+3, 2024)])
        st.text(upper)
        category = st.selectbox('select category', options=['All','Cat1', 'Cat2', 'Cat3','Cat4','Cat5', 'Cat6'])
        st.radio('select', [''])
        
    # function to add data to the dictionary
    def add_record(data_dict, input, time, score, sentiment):
        new_index = len(data_dict) +1
        data_dict[f'{new_index}'] = {'review':input, 'timestamp':time,'score':score,'sentiment':sentiment}
    
    
    # DATA PREPROCESSING    
    sample = pd.read_csv('sample.csv')
    
    sample['Timestamp'] = sample['Timestamp'].astype('datetime64')
    sample = sample[sample['Timestamp'] >= str(lower)]
    sample = sample[sample['Timestamp'] <= str(upper)]
    
    # create a  column for random categories
    category  = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5', 'Cat6']
    random = np.random.choice(category, size=len(sample))
    sample['Category'] = random
    
    positive = sample[sample['Sentiment'] == 'positive']
    neutral = sample[sample['Sentiment'] == 'neutral']
    negative = sample[sample['Sentiment'] == 'negative']

    
    
    # creating the dictionary of reviews
    if 'list_of_reviews' not in st.session_state:
        st.session_state.list_of_reviews={'sample':'neutral'}
        st.session_state.review_dataset = {'1': {'review': 'placeholder', 'timestamp': '2022-03-30', 'score': 0.7, 'sentiment': 'Positive'}}


    # input section
    with col[2].form(key='input form'):
        user_input = st.text_input('Please leave a review')
        
        button = st.form_submit_button(label='Submit')
        
        if button:
            
            # sentiment analyzer
            encode_text = tokenizer(user_input, return_tensors='pt')
            output = model(**encode_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            scores_dict = {
                'negative' : scores[0],
                'neutral' : scores[1],
                'positive' : scores[2]
            }
            max_key = max(scores_dict, key=scores_dict.get)
            max_value = scores_dict[max_key]
            
            new_sentiment = max_key
            st.session_state.list_of_reviews[user_input] = new_sentiment
            
            #create dataset
            timestamp = datetime.datetime.now()
            add_record(st.session_state.review_dataset, user_input, timestamp,max_value, max_key)


    

    #reviews section
    col[0].subheader(':blue[Reviews]', divider='rainbow') 
    col[0].write(f'Review Count: {len(st.session_state.list_of_reviews)}')  
    for index, row in sample.head(5).iterrows():
        
        color = text_color(row['Sentiment'])   
        colored_text = f'<span style="color:{color}">{row["Sentiment"]}</span>'
        res = f'* {row["Summary"]} \[{colored_text}\]'
        col[0].write(res, unsafe_allow_html=True)


    # <------------------------Plotting the bar chart-------------------------->
    st.set_option('deprecation.showPyplotGlobalUse', False) 
    remark_list = list(st.session_state.list_of_reviews.values())

    value_counts = pd.Series(remark_list).value_counts().reset_index()
    value_counts.columns = ['Customer Remark', 'Count']
    # Plot bar chart
    plt.figure(figsize=(8,6))
    sns.barplot(x='Customer Remark', y='Count', data=value_counts)
    plt.ylabel('Count')
    plt.xlabel('Customer Remark')
    plt.title('Value Counts of Customer remark')
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    col[1].pyplot()
    

    
    value_counts = sample['Sentiment'].value_counts()

    # <-----------------------------Plot pie chart------------------------------->
    plt.figure(figsize=(5, 3))
    colors = ['lightgreen', 'orange', 'red'] 
    fig, ax = plt.subplots(figsize=(5, 4))
    outer_circle = ax.pie(value_counts, labels=value_counts.index, colors=colors, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    inner_circle = ax.pie([1], radius=0.46, colors=['white'])
    # Display count and percentage on each segment
    legend_labels = [f"{category}: {count} ({value_counts[category] / len(value_counts) * 1:.1f}%)" for category, count in zip(value_counts.index, value_counts)]

    plt.axis('equal')
    plt.title('COUNT OF SENTIMENT') 
    col[1].pyplot(fig)
    
    # <-----------------------------Plotting the Stacked BarChart------------------------------->
    pivot_table = pd.pivot_table(sample, values='Id', index='Category', columns='Sentiment', aggfunc='count')
    stacked_df = pivot_table.reset_index()

    df1 = stacked_df.copy()

    # Create a stacked bar chart
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    plt.bar(df1['Category'], df1['neutral'],width=0.5, bottom=df1['negative'], label='Neutral', color='orange')
    plt.bar(df1['Category'], df1['positive'], width=0.5,bottom=df1['negative']+df1['neutral'], label='Positive', color='green')
    plt.bar(df1['Category'], df1['negative'], width=0.5,label='Negtative', color='red')

    # Add a title and labels
    plt.title('Distribution of sentiment by Category')
    plt.xlabel('Category')
    plt.ylabel('Count (Log Scale)')
    plt.yscale('log')

    plt.legend()
    plt.show()
    col[2].pyplot(fig3)
    
    # <-----------------------------Plotting the area chart------------------------------->

    # Group by month
    sales_by_month = sample.groupby(sample['Timestamp'].dt.to_period('M')).count()
    positive_sales_by_month = positive.groupby(positive['Timestamp'].dt.to_period('M')).count()
    negative_sales_by_month = negative.groupby(negative['Timestamp'].dt.to_period('M')).count()
    neutral_sales_by_month = neutral.groupby(neutral['Timestamp'].dt.to_period('M')).count()

    fig, ax = plt.subplots(figsize=(10, 4))

    plt.yscale('log')
    
    plt.plot(sales_by_month.index.to_timestamp(), sales_by_month['Id'],linewidth=1, color='skyblue', linestyle='--', label='Total Sales')
    plt.fill_between(sales_by_month.index.to_timestamp(), sales_by_month['Id'], color='skyblue', alpha=0.3)

    # Plot positive sales line
    plt.plot(positive_sales_by_month.index.to_timestamp(), positive_sales_by_month['Id'], linestyle='--',linewidth=0.5, color='green', label='Positive Review')
    plt.fill_between(positive_sales_by_month.index.to_timestamp(), positive_sales_by_month['Id'], color='green', alpha=0.3)

    # Plot negative sales line
    plt.plot(negative_sales_by_month.index.to_timestamp(), negative_sales_by_month['Id'], linestyle='--',linewidth=0.5, color='red', label='Negative Review')
    plt.fill_between(negative_sales_by_month.index.to_timestamp(), negative_sales_by_month['Id'], color='red', alpha=0.3)

    # Plot neutral sales line
    plt.plot(neutral_sales_by_month.index.to_timestamp(), neutral_sales_by_month['Id'], linestyle='--',linewidth=0.5, color='orange', label='Neutral Review')
    plt.fill_between(neutral_sales_by_month.index.to_timestamp(), neutral_sales_by_month['Id'], color='orange', alpha=0.3)

    # Set labels and title
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.title('MONTHLY NUMBER OF REVIEWS BY SENTIMENT')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    
    
if __name__ == "__main__":
    main()