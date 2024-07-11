import pandas as pd  #pandas is sued for data manipulation
import numpy as np   #used for numerical compution
import streamlit as st  #this is python library used for creating web applications using simple python syntax
import matplotlib.pyplot as plt   #used for creating visualizations in python
import seaborn as sns   #for creating informative and attractive statistical graphics
import plotly.express as px  # generating complex visualizations with minimal code
import plotly.graph_objects as go  #provides a flexible and powerful interface for constructing various types of plots and figures. 
from io import StringIO  #StringIO is a class from the io module in Python. It provides a way to treat strings as file-like objects.
from wordcloud import WordCloud  #for creating wordcloud
import logging
import base64
from sklearn.model_selection import train_test_split #this helps in splitting the data into traing and testing set for model development
from sklearn.linear_model import LinearRegression #use to model relationship between a dependent and an independent variable
from sklearn.metrics import r2_score #variability between the dependent and independent variable. 0 means independent variable do not explain any variability in dependent variable. 1 means independent variable explains all variability in dependent variable..Higher score indicates model better fits the data
from sklearn.linear_model import LogisticRegression  #modelling the relationship between a binary dependent varibale and one or more independent variable
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
#accuracy is used to evaluate the performance of classification model. How correctly the model has predicted the values
#classification report provides insights into the performance of classification model for each class
#confusion medics provides a report by comparing actual class with the predicted class
from sklearn.neighbors import KNeighborsClassifier #supervised learning. memorizes entire training data set andmake predictions based on similarity between new instances and exisiting instances
from sklearn.tree import DecisionTreeClassifier,plot_tree #supervised learning.heirarchial structure. determine how to split input data at each node.nide represnets decsion . each node or decision is split into another dnodes or decisison. Leaf node at the end represent the final decision
from sklearn.cluster import KMeans #unsupervised learning. Partituoning dataset into number of clusters
from sklearn.ensemble import RandomForestClassifier  #insted of amking one big decision tree this algorithmn craetes multiple trees and combine the results of all the trees. These many decision trees are called as forrest. It randomly sleects subset of features to make trees
from sklearn.metrics import silhouette_score  #a metric to measure the accuracy of the decsion tree. High score means that daat is correctly placed in its cluster whereas negative value indicates it ahs been placed to teh wrong cluster
import subprocess    #To connect to other python page by redirecting it
import io
from textblob import TextBlob#for sentiment analysis
from PIL import Image #foe loading images
from fpdf import FPDF #to generated pdf
import os #interacting with operating system
import tempfile

#to make the display on the output wide we mneed to change the configuration of the page
st.set_page_config(layout="wide")


def get_base64_of_bin_file(bin_file): # a function that takes input as bin_file which is the finary file
    with open(bin_file, 'rb') as f: #opens binary file in te binary mode. with ensures closing teh file after operations
        data = f.read() #reads the content of the file and stores it in the data variable
    return base64.b64encode(data).decode()  #converts the binary data into a Base64 encoded byte string.It decodes the Base64 encoded byte string into a UTF-8 string.decode is necesssary because b64 returns byts and we want string

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) #calls the function get_base64_of_bin_file(png_file) to convert the PNG image file specified by png_file into its Base64 encoded representation.
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Use this function to set the background image of app
set_png_as_page_bg('bg2.png')



#setting the tutle for the webpage
st.title(':white[CipherQuest]')
# This name means embark on the quest to decode the hidden insight in data

img=Image.open("C:\\Users\\DELL\\Downloads\\logo2.png")  #loads the image into a PIL Image object named img.
wid=350 #width pf the image
heigh=1000  #height of the image
img_resize=img.resize((heigh,wid))  #resizing the image with the avove mentioned parameter
st.image(img_resize,use_column_width=True)  #displaying that image


#creating a button for dashboard
#on clicking the button the user will be navigated to the dashboard creation page
if (st.button("Create Dashboard")):
    subprocess.run(["streamlit", "run", "dashboard.py"],shell=True) 
#subprocess.run() is a function from the subprocess module in Python, which allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.
#The function subprocess.run() takes a list of strings as its first argument. In this case, ["streamlit", "run", "dashboard.py"] is the list of command-line arguments to be passed to the subprocess.
#"streamlit" is the name of the command or program to be executed.
#"run" is an argument passed to the streamlit command, indicating that it should run a Streamlit application.
#"dashboard.py" is the name of the Python script containing the Streamlit dashboard application that you want to run.
# shell tells the python to run the code as if we r running it in terminal or command prompt


# Creating input for the link by the user for the dataset
link = st.file_uploader("Upload the file: ", type=["xlsx", "csv"])


#creating a function to open the dataset which will be used further for analysis in the entire app
def load_data(file): 
    if link.name.endswith(".xlsx"): #check if the file ends with xlsx
        data = pd.read_excel(file)  #if yes read that file 
        return data  #and return the results
    elif link.name.endswith(".csv"):  #else check if the file ends with csv
        data = pd.read_csv(file) #if yes read that file
        return data  #and return the file

#creating a function to drop column
def drop_col(df,columns1): #actual dataframe and columns that will be specified by the user will be the parameters for this function
    for col in columns1: #if the user has selcted more than one column then iterate through each in the list of the columns
        df.drop(columns=col,inplace=True) #drop those columns specified by the user
    st.write(df)    #then print the dataframe with the dropped columns



#create a function to replace na with mean
def replace_mean(df,columns):  #actual dataframe and columns that will be specified by the user will be the parameters for this function
    for col in columns:#if the user has selcted more than one column then iterate through each in the list of the columns
        col_mean = df[col].mean() #create a new variable that stores the mean of the column specified by the user
        df[col].fillna(col_mean, inplace=True) #using the fillna function to replace the blank vaues with the above calculated mean value and inplace indicates that the chnages are made to the actual dataset

    st.write(df)  #display the dataset after dropping the values for user to see the changes


#creating a function for replacing na with median
def replace_median(df,columns):  #actual dataframe and columns that will be specified by the user will be the parameters for this function
    for col in columns:#if the user has selcted more than one column then iterate through each in the list of the columns
        col_median = df[col].median()  #create a new variable that stores the median of the column specified by the user
        df[col].fillna(col_median, inplace=True) #using the fillna function to replace the blank vaues with the above calculated median value and inplace indicates that the chnages are made to the actual dataset

    st.write(df) #display the dataset after dropping the values for user to see the changes



#create a function for replacing na with a default value by the user
def replace_default(df,columns,val): #actual dataframe,value(by user) and columns that will be specified by the user will be the parameters for this function
    for col in columns: #if the user has selcted more than one column then iterate through each in the list of the columns
        df[col].fillna(val,inplace=True)  #using the fillna to fill the na values with the default value that will be specified by the user
    st.write(df)  #display the dataset after dropping the values for user to see the changes



#creating a function to filter the dataset as per the categorical value for a specific column specified by the user
def filter(df,column,val): #the actual dataframe, column specified by user and the value acc to which the datatset is to be filtered will be the parameters of this function
    df_filter=df[df[column].isin(val)]  #filter the dataset as per the above condition and store that filtered dataset with new name
    st.write(df_filter) #display that dataset
    st.write("Dimension: ")  #also specifying the dimensions of the new filtered dataset
    st.success(df_filter.shape)

#creating a function to filter the dataset based on the values of the column that are greater than the user specified value
def greater_record(df,column,value): #the actual dataframe, column specified by user and the value acc to which the datatset is to be filtered will be the parameters of this function
    df_filter=df[df[column]>value] #create a new dataframe that will have the filtered values
    st.write(df_filter) #display that dataset
    st.write("Dimensions:")  #also displaying the dimension of this new datafarme craeted
    st.success(df_filter.shape)

#creating a function to filter the dataset based on the values of the column that are less than the user specified value
def lessthan_record(df,column,value): #the actual dataframe, column specified by user and the value acc to which the datatset is to be filtered will be the parameters of this function
    df_filter=df[df[column]<value] #create a new dataframe that will have the filtered values
    st.write(df_filter) #display that dataset
    st.write("Dimensions:") #also displaying the dimension of this new datafarme craeted
    st.success(df_filter.shape)

#function for splitting the column
def split_col(df, cols, delimiter): #create a function
    for col in cols:  #iterate through the list of columns
        split_columns = df[col].str.split(delimiter, expand=True) #split the column by delimeter. The expand function splits it into different column each
        split_columns.columns = [f"{col}_{i}" for i in range(split_columns.shape[1])] #reanming the columns by iterating through the size of split list
        df = pd.concat([df, split_columns], axis=1) #concatenate this splits to original datafarme
    st.write(df.head())  #display the dataset to verify


#function for column renaming
def col_rename(df,old_name,new_name): #creating a function for renaming column with the following parametrs
    old_name = list(old_name)
    new_name = list(new_name)
    col_rename_dict = {}  # Initialize an empty dictionary to store old-to-new name mappings
    # Iterate over old_names and new_names simultaneously
    for old, new in zip(old_name, new_name):
        col_rename_dict[old] = new  # MapPING each old name to its corresponding new name
        st.write(f"Mapping '{old}' to '{new}'.")
    df.rename(columns=col_rename_dict, inplace=True)  # Renaming the columns
    st.write(df.head()) 


def save_plot_as_jpeg(plot, plot_title):  #ceates a functuon that takes 2 paarmetres one is the pot to be displayed and othe is the name for the iamge file
    tmpfile_path = f"{plot_title}.jpeg"  #craetes a fil e anme for the temproy plot image. like it varies as the input by the user
    plot.savefig(tmpfile_path, format='jpeg')  #saves the plot to the disk in jpeg format
    return tmpfile_path  #retuns the saved jpeg image
    
def get_download_link(file_path, file_label='Download Plot'):  #craetes a function for downloading the plot takes 2 paarmetres one is the path hwre it is to be downloaded and the other is the link name for downloading
    with open(file_path, "rb") as f: #opens the file i the binary mode and with function c,oes the file after operations
        bytes_data = f.read() #reads the file and stores it in a new variable
    b64 = base64.b64encode(bytes_data).decode()  #this encodes the binary data of the above file into bytes and then the decode fucntion decodes it to UTF-8 string
    href = f'<a href="data:image/jpeg;base64,{b64}" download="{os.path.basename(file_path)}">{file_label}</a>'
    # Specifies that the href attribute contains a Base64 encoded image.it represnts the MIME(Multipurpose Internet Mail Extensions) type of the data.specifies anture and format of the image shared over internet.MIME types are used to specify the content type of data being sent in HTTP headers.consitts of media tpe and subtype seperated y slash
    # Inserts the Base64 encoded data.
    #Specifies the filename to be downloaded. The download attribute tells the browser to download the linked file instead of navigating to it.
    #Inserts the label text for the download link.
    return href#return thr link craeted in the above line



#creating functions for graph creation

#function for barplot
def create_barplot(df, x_column, y_column,x_axis,y_axis ,title):
    fig = px.bar(df, x=x_column, y=y_column, title=title) #using the plotly function to create the bar chart which has parameters like dataframe, x column , y column and the title
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #using update_layout add the labels for x and y axis
    return fig #returing the figure so that if gets displayed

#function for countplot
def create_snsplot(df, x_column,x_axis,y_axis, title): #creating a function with the desired parameters
    plt.figure(figsize=(18, 18)) #giving the dimensions to the plot
    fig = sns.countplot(df,x=x_column,palette="viridis")  #creating a countplot using seaborn library and following parameters
    plt.title(title)  #giving a suitable user defined title to plot
    plt.xlabel(x_axis)  #label for xaxis
    plt.xticks(rotation=45) #rotate the x axis label so that it does not get conjusted for long names
    plt.ylabel(y_axis) #label for y axis
    for p in fig.patches:
            fig.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom')
    return plt  #return the plot to display

# function for scatter plot
def create_scatterplot(df,x_column,y_column,x_axis,y_axis,title):
    fig=px.scatter(df,x=x_column,y=y_column,title=title)  #using the px function to craete the scatterplot with parameters like numerical x and y variables and the title for the chart
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #using update_layout for teh x and y lables by the user
    return fig  #retruning the chart so taht it gets dispalyed

# function for line plot
def create_lineplot(df,x_column,y_column,x_axis,y_axis,title):
    fig=px.line(df,x=x_column,y=y_column,title=title)  #using the px lineplot function o display the line chart with x and y column and the title for the chart thata re specified by the user
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #giving x and y label for the chart
    return fig  #returning the figure so taht it egts dispalyed

#creating a function for histogram
def create_histogram(df,x_column,y_column,x_axis,y_axis,title):
    fig=px.histogram(df,x=x_column,y=y_column,title=title)  #using the px function to craete the histogram with the x and y variable as the parametrs as well as the title defined by the user
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #giving user specified x and y label
    return fig  #returning the figure so that it gets displayed


#creating a function for bubble chart
def create_bubbleplot(df,x_column,y_column,size_1,color_1,x_axis,y_axis,title):
    fig=px.scatter(df,x=x_column,y=y_column,size=size_1,title=title,color=color_1)  #using the px method to craete the bubble chart which takes 2 variables x and y and another variable which specifies the size of each bubble an dthe title specified by teh user
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #giving labels to x and y axis
    return fig  #returing the figure so taht it gets displayed



#creating a function for voilin chart
def create_violin_plot(df,x_column,y_column,x_axis,y_axis,title):
    fig=px.violin(df,x=x_column,y=y_column,title=title)  #using the px function to craete the voilin chart that takes the paarmeter as datafraem, x and y varobales and title by the user. the voilin plot helps in funding the density of each object
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #giving x and y labels specified by the user
    return fig  #returing the figuer so that it gets displayed


#creating a function for gantt chart
def gantt_chart(df,start,finish,y_column,x_axis,y_axis,title):
    fig=px.timeline(df,x_start=start,x_end=finish,y=y_column,title=title)  #using the px timeline function for creating a gantt chart that are mostly use to see the project deadline. it takes parameters like actual value column the target value column title and the y variable for seperationas per categories
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #giving labels for x and y axis
    return fig  #returning teh figure so that it egts displayed


#creating a function for boxplot
def box_plot(df,x_column,y_column,x_axis,y_axis,title):
    fig=px.box(df,x=x_column,y=y_column,title=title)  #using the px method to craete a boxplot for numerical summarization. it takes parameters like datframe, x and y variable where the y avriable is numerical and tutle specified by teh suer
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #giving labels to x and y axis
    return fig  #returning teh figure so taht it gets displayed


#creating a function for wordcloud
def create_wordcloud(df,x_column,col,title):
    column=' '.join(df[x_column].astype(str))  #taking all the comments or text and joining it to create a single string out of it and store it in one variable
    wordcloud=WordCloud(width=500,height=500,background_color=col).generate(column)  #using the wordcloud method to craete wordcloud , specifying width and neight and background color as specified by user and then by using teh generate method creating the wordcloud
    plt.figure(figsize=(10, 10)) #giving the dimensions for the figure
    plt.imshow(wordcloud, interpolation='bilinear')  #displaying wordcloud
    plt.title(title) #title as specified by teh suer
    plt.axis('off')  # Turn off axis
    return plt  #returing the figure so that it gets dispalyed


#creating a function for 3d  line plot
def lineplot_3d(df,x_column,y_column,z_column,title):
    fig=px.line_3d(df,x=x_column,y=y_column,z=z_column,title=title) #craeting a 3d line chart with 3 diemsnions so it takes 3 variables and a title bu the user
    return fig  #returning thr figure so that it gets dispalyed


#creating a function for 3d scatter plot
def scatterplot_3d(df,x_column,y_column,z_column,color,title):
    fig=px.scatter_3d(df,x=x_column,y=y_column,z=z_column,color=color,title=title)  #craeting a 3d scatter plot so it takes 3 variables and a title specified by the user
    return fig   #returining teh figure so taht it gets displayed


#creating a function for distplot
#visualize the distribution of a single variable in a dataset. suppose height of people, it would create a barplot to see ferquency of each data point and also a smooth cureve that shows wheter data is symmetric or skewed
def dist_plot(df,x_column,title):
    fig=sns.displot(df[x_column],kde=True)  #using the seaborn displot function to craete a distplot by taking only one parameter that is the x variable
    plt.title(title)  #title by the user
    return fig  #returing teh figure so taht it gets displayed


#creating a function for jointplot
#visualize the relationship between two variables.eg there are 2 variables height and weight of the people and this plot creates a scatter plot to see relationship between them and also histogram to see their individual distribution
def joint_plot(df,x_column,y_column,title):
    fig=sns.jointplot(df,x=x_column,y=y_column)  #suing teh seaborn jointplot function to craete a jointplot. it takes 2 variables
    plt.title(title)  #title by the user
    plt.xticks(rotation=45)
    return fig   #returing teh figure so taht it gets displayed

#creating a function for pairplot
#this creates multiple scatter plots for each and every feature in the dataset to quickly see if any relation exist between any two variables
def pair_plot(df,title):
    fig=sns.pairplot(df)  #using the seaborn pairplot function to craete multiple plots from the dataset. it takes only the dataset and teh hues as the parameter
    plt.title(title)  #title by the user
    return fig  #returing teh figure so taht it gets displayed

def create_piechart(df, val, names_1, title):
    fig = px.pie(df, values=val, names=names_1, title=title) 
    return fig

#creating a function for kde plot
#to visualize the distribution of a dataset.KDE plot as creating a smooth outline of the data.KDE plot smoothes out the jaggedness of individual data points and gives you a nice, smooth curve that shows you where most of the data is concentrated and how it spreads out across different values. 
def kde_plot(df, x_column, y_column, title): #this line creates a function and takes the following arguemnets as inputs
    fig, ax = plt.subplots() #this creates subplots. the fig represents the entire figure and the ax represents the axis to plot the figure
    sns.kdeplot(data=df, x=x_column, y=y_column, ax=ax) #creates a kde plot showing dostribution of datapoints in x and y axis
    ax.set_title(title)  #gives a title to the plot
    plt.figure(figsize=(6,6))
    return fig  



#creating a function for regression plot
#creating a scatter plot alog with a line that shows the trend in the dataset
def reg_plot(df,x_column,y_column,title):
    plt.figure(figsize=(4, 4))
    fig=sns.lmplot(df,x=x_column,y=y_column,height=9)  #creating a regression plot that shows correlation and between the vraiables. it takes 2 variables x and y
    plt.title(title)   #title by the user
    plt.tight_layout()
    return fig  #returing teh figure so taht it gets displayed

#creating a function for heatmap
# Heatmaps are often used to identify patterns, trends, or correlations within large datasets, making complex information easier to understand at a glance. 
def create_heatmap(df):
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns  # Select only numeric columns
    df_corr = df[numeric_columns].corr()  # Calculate correlation of numeric columns
    fig, ax = plt.subplots()  # Create a figure and axis object
    sns.heatmap(df_corr, ax=ax)  # Create heatmap using seaborn with the specified axis
    return fig


#creating a function for comparing 2 bars in a plot(stacked bar chart)
def stacked_bar(df,x_column,y1_column,y2_column,title):
    fig=px.bar(df,x=x_column,y=[y1_column,y2_column],title=title)  #creaing a stacked bar plot where it takes parameters like the x variable for category seperation and 2 y variables for stacking teh columnsa nd the title by teh suer
    return fig   #returing teh figure so taht it gets displayed


#creating a function for comparing 2 lines in a plot
def double_line(df,x_column,y1_column,y2_column,title):
    fig=px.line(df,x=x_column,y=[y1_column,y2_column],title=title)  #using teh px line function to craete a line chart for 2 variables simultaneoudlu for comaprison. It takes one x variable and 2 y avriable and a title for the cahrt
    return fig   #returing teh figure so taht it gets displayed
  

#creating a function for candlestick chart
def candlestick_chart(df,x_column,open,high,low,close):
    fig = go.Figure(data=[go.Candlestick(x=df[x_column],  #using graph objects to craeet a candlestick chart. x axis remanins the time period
                    open=df[open],  #the opening price for thr stock
                    high=df[high],  #the high value for the stock
                    low=df[low],  #the low value for teh stock
                    close=df[close])])  #the close price for teh stock
    return fig  #returing teh figure so taht it gets displayed


def sentiment_analysis(df, col):  #creating a function to find sentiments in the text
    if isinstance(col, str):  #check if the column is the string or no
        sentiments = []  #creating a blank list to append the sentiment values that are calculated
        for text in df[col]:  #go through each value of the column
            blob = TextBlob(text)  #create a textblob object for the text for better analysis
            polarity = blob.sentiment.polarity  #calculating polarity score for sentiment analysis
            # Categorize sentiment into positive, negative, or neutral
            if polarity > 0:
                sentiments.append("Positive")
            elif polarity < 0:
                sentiments.append("Negative")
            else:
                sentiments.append("Neutral")
        df['Sentiment'] = sentiments  #adding a new column to dataset to add the sentiment values
        sentiment_counts = df['Sentiment'].value_counts()  #calculate the value count of this sentimenets
        plt.figure(figsize=(8, 6))  #adjusting the size of the figure
        fig=sns.countplot(x='Sentiment', data=df) #craeting a count plot
        plt.title('Sentiment Analysis') #titl for the chart
        plt.xlabel('Sentiment')  #labels for x axis
        plt.ylabel('Count') #labels for y axis
        plt.xticks(rotation=45)
        for p in fig.patches:
            fig.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom')  #rotating the labels to better fit and visibility
        return df, sentiment_counts,plt

#creating a function for linear regression
#this is used to find the relationship between the dependent ad the independent variable in the dataset
def linear_regression(df, x_columns, target_column, t_size):  #creates a function that takes 4 parameters i.e dataframe, features column(x col),target column an dthe test_size for splitting the dataset into train and test
    model = LinearRegression() #create an instance of the LR model
    X = df[x_columns].values #extracts the feature variables and converts them into numpy array,for compatible with model
    y = df[target_column].values  #extracts the values of target variable and converts them into array
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42) #splits the dataset into training and testing set with the size of the test set specified by the user. the random state ensures reproducilibility of the split
    model.fit(X_train, y_train)  #this line fits the traiing data to the model to train it.
    y_predict = model.predict(X_test)  #this lie uses the above trained model to make preidictions on the testing data
    st.write("The prediction is:")  #this print this tsatemnet
    df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})  #this creates a dataframe that displays the ctual values (y_test) along with the predicted values by the model to match it
    st.write(df_new) #print this dataframe
    r2_s = r2_score(y_test, y_predict) #r2 score calculates proportion of variance in the target variable that is predicted.Assess goodness of fit of the regression model to testing data.Higher value indicates good fit.
    st.write("The R2 score for this model is:", r2_s)
    st.write("Intercept:", model.intercept_)  #intercept represents the value of the target variable when all the feature variables are zero
    st.write("Coefficients:", model.coef_)  #coefficient indicate the change in the target variable for one unit change in each feature variable


#craeting a function for logistic regression
#this algorithmn finds relationship between an independent varaible and a binary dependent variable
def logistic_regression(df, x_column, target_column, t_size):  #creates a function for logistic reression that takes the following arguemnets as input to it
    model = LogisticRegression()  #creating instance of logistic regression model
    x = df[x_column].values  ##extracts the feature variables and converts them into numpy array,for compatible with model
    y = df[target_column].values #takes the values of the target variable an dconverts them into array
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=42) #splitting the dataset into training an dtesting set with size of the testing set specifies by the user and random state indicates the reproducilibility of the test
    model.fit(x_train, y_train)  #fitting the model with the training data to train it
    y_predict = model.predict(x_test)   #this lie uses the above trained model to make preidictions on the testing data
    st.write("The prediction is:") #print this statemnet
    df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})  #craetes a dataframe with the actual value of the dataset testing part(y_test) and the predicted value by the model for comparison
    st.write(df_new) #print the dataframe
    acc_score = accuracy_score(y_test, y_predict)  #checks the accuracy of the model in prediction. it compares the true values(y_test) with predicted values and returns accuracy score i.r perecntage of correctly predicted value to the total number of instances in testing dataset.higher value better
    st.write("The accuracy score for this model is:", acc_score)  #print it
    

#creating a function for k-nearest neighbor
# to predict the class of a new data point, the algorithm looks at the 'k' closest data points in the training set. It assigns the majority class among these neighbors as the predicted class for the new data point
def knn_algo(df,x_column,y_column,t_size,neighbor_size): #creating a function for knn algorithmn with the follwoing parametres as input to this function
        x=df[x_column].values  #taking the values of this feature column and converting them into array for easy compatibility by the model
        y=df[y_column].values  #taking the values of this target vraiable and converting to array
        model=KNeighborsClassifier(n_neighbors=neighbor_size)  #creates an instance of kneighborsclassifier and specify the number of neighbors
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=t_size,random_state=42) #split the dataset into training and testing sets with teh size of the test data specified by the user
        model.fit(x_train,y_train) #training the model on the training data
        y_predict=model.predict(x_test)  #this lie uses the above trained model to make preidictions on the testing data
        st.write("The prediction is: ")  #print this
        df_new=pd.DataFrame({'Actual':y_test,'Predicted':y_predict})  #create a dataframe of the actual values of the test data (y_test) and the predicted values by the model
        st.write(df_new) #print it
        acc_score=accuracy_score(y_test,y_predict)  #calculate the accuracy score of the model in predicton. it compares true values iwth teh predicted values and returns accuracy score of the model .perecntage of correctly predicted value to the number of instances in training set  
        st.write("The accuracy score for this model is:", acc_score)  #print the value

        # Plotting
        plot_data = np.hstack((x_test, y_test.reshape(-1, 1), y_predict.reshape(-1, 1))) #horizotally stacks xtest and yest and y predict . reshapes into 1D array so that they have same shape
        #so plot_data becomes 2D as it has rows with xtest ytest and ypredict
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_data[:, 0], plot_data[:, 1], color='blue', label='Actual') #plot[:,0] reprsents feature column and plot[:,1 ] repsents the actual target value
        plt.scatter(plot_data[:, 0], plot_data[:, 2], color='red', label='Predicted') #plot[:,0] reprsents feature column and plot[:,1 ] repsents the  predicted value
        plt.xlabel(x_column)  #provide the appropriate lables
        plt.ylabel(y_column)
        plt.title('KNN Scatter Plot')
        plt.legend()  #legend to distinguish between the points
        plt.grid(True) #add grids for better readibility
        st.pyplot(plt)#displaying plot




#craeting a function for decision tree
        #decision tress splits the input or the decioson into further nodes or decision and it goes on further till the point where the node cannot be split further. that is it reachesthe leaf nodes
def decision_tree(df,x_column, target_column, t_size,max_depth): #this creates a function to craete a decision tree and takes the fllowing parameter as the input
    model = DecisionTreeClassifier(max_depth=max_depth)  #creates a instance of the decisiontreeclassifier model and maximum depth specifies the lmaximum level of the tree to prevent it from being to complex
    x = df[x_column].values #takes the values of the feature column and convert it into the array
    y = df[target_column].values  #takes the values of the target column an dconverts it into array for model compatibility
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=42)  #splits the data into training and testing set with teh size of teh testing set specifiedby teh user
    model.fit(x_train, y_train)  #train the model with the training dataset
    y_predict = model.predict(x_test)  #use the above trained model to predict the value of the etsting data
    st.write("The prediction is:")  #print this statement
    df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})  #craete a dataframe with the actual values y_test and the predicted value by the model
    st.write(df_new)  #print it
    acc_score = accuracy_score(y_test, y_predict)#calculate the accuracy score of the model in predicton. it compares true values iwth teh predicted values and returns accuracy score of the model .perecntage of correctly predicted value to the number of instances in training set  
    st.write("The accuracy score for this model is:", acc_score)  #print the accuracy_score
    class_rep=classification_report(y_test,y_predict)  #this generates metrice slike the precision score,recall,f1core.this is use to get performace of model
    st.write("Classification Report :",class_rep)
    conf_matrix=confusion_matrix(y_test,y_predict) #showing the counts of true positive, true negative, false positive, and false negative predictions.
    st.write("Confusion Matrix: ",conf_matrix)
    #precision= It calculates the ratio of correctly predicted positive observations to the total predicted positive observations
    #recall=  It calculates the ratio of correctly predicted positive observations to the total actual positive observations.
    #F1-score = It provides a balance between precision and recall, as it takes into account both false positives and false negatives. F1-score reaches its best value at 1 (perfect precision and recall) and worst at 0. F1-score is useful when the class distribution is imbalanced.
    #supprot = Support is useful for understanding the distribution of actual instances across different classes
    plt.figure(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=x_column, class_names=model.classes_.tolist())  #plots a decison tree with teh model which is trained above, filled specifies that the nodes should be filled with colors. Features name provides the name of the features used in the model.class_name specifies the names of the targets in the model
    st.pyplot(plt)

    

def decision_tree_for_cat(df, x_columns, target_column, t_size, max_depth):  #craete a function
    model = DecisionTreeClassifier(max_depth=max_depth)  #an instance of the model
    X = df[x_columns]  # Use multiple columns for features  
    y = df[target_column]  #target column
    # One-hot encode categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns  #cahcek if the columns in the category are object dtype
    if len(categorical_columns) > 0: #if categorical column is presnet
        encoder = OneHotEncoder(handle_unknown='ignore')  #handle_unknown='ignore', which means it will ignore unknown categories during transformation.
        X_encoded = encoder.fit_transform(X[categorical_columns])#encoder is then fit to the categorical features (X[categorical_columns]) and transforms them into a one-hot encoded format (X_encoded).
        X_encoded = pd.DataFrame(X_encoded.toarray())  # Convert to DataFrame to use for analysis
        X = pd.concat([X.drop(columns=categorical_columns), X_encoded], axis=1)  #from the x , drop categorical columns and add the one hot ecoded dataframe
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42) #split into train and test set
    model.fit(x_train, y_train) #train the model
    y_predict = model.predict(x_test) #prediction based on the above trained model
    st.write("The prediction is:") #print this
    df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict}) #new dataframe of actual and predicted value
    st.write(df_new) #print it
    acc_score = accuracy_score(y_test, y_predict) #get accuracy score of teh model
    st.write("The accuracy score for this model is:", acc_score) #print it
    #one hot encoding is the method of converting categorical value to numerical by assigning 1 and 0 and craeting muktiple columns for it

#creating a function for kmeans
    #this algorithmn divides the datapoints into different group, where each group contains points similar to each other with respect to features. it continues iteratively untill the cluster size stabalizes
def k_means_clustering(df,x_column,n_clust):  #craetes a functiom for making k means clusters with the following parameters
    x_col=df[[x_column]]  #extracts the following column from the datset
    kmeans=KMeans(n_clusters=n_clust) #initializes the model and specifies the number of clusters
    kmeans.fit(x_col)  #fits the model to the data. it computes clusters centers and assign each point to the nearesr cluster based on eucleidean distance
    clusters=kmeans.predict(x_col) # predicts the cluster labels for each data point based on the fitted KMeans model
    df['Clusters']=clusters#This line adds a new column named 'Clusters' to the original DataFrame df and assigns the predicted cluster labels (clusters) to it
    st.write(df) #print the dataframe
    #since there is no accuracy score metric for clustering algorithmn as it is an unsupervised type of ML with no ground label truhs
    #so we can use other parameters like silhouette_score to measure its saccuracy
    #positive high score indicates that the samples are assigned to their own clusters whereas negative score indicates taht samples have been assignmed to wrong clusters
    silhouette_s = silhouette_score(df[[x_column]], df['Clusters'])  #to measure if the clusters are assigned to thair correct clsuters
    st.write("Silhouette Score:", silhouette_s) #print it



#creating a function for Random Forest
#random forest means suppose we ahve to make a big decision and we want advice from various experts. so random forest creates multiple decsision tere and combine tehir answers to amke a better decision/prediction
#rgis only takes numerical values in independent variable
def random_forest(df,x_col,target_column,t_size): #create a function for craeting random forest model
    model=RandomForestClassifier()  #initialize the model. Create an instnace of teh model
    x=df[x_col].values #takes the values of the feature column and convert it into the array
    y=df[target_column].values  #takes the values of the target column an dconverts it into array for model compatibility
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=t_size,random_state=42) #splits the data into training and testing set with teh size of teh testing set specifiedby teh user
    model.fit(x_train,y_train)#train the model with the training dataset
    y_predict=model.predict(x_test)  #use the above trained model to predict the value of the etsting data
    st.write("The prediction is:") #print this statemnet
    df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict}) #craete a dataframe with the actual values y_test and the predicted value by the model
    st.write(df_new) #print the dataframe
    acc_score = accuracy_score(y_test, y_predict) #calculate the accuracy score of the model in predicton. it compares true values iwth teh predicted values and returns accuracy score of the model .perecntage of correctly predicted value to the number of instances in training set  
    st.write("The accuracy score for this model is:", acc_score)
    class_rep=classification_report(y_test,y_predict)#this generates metrice slike the precision score,recall,f1core.this is use to get performace of model
    st.write("Classification Report :",class_rep)
    conf_matrix=confusion_matrix(y_test,y_predict) #showing the counts of true positive, true negative, false positive, and false negative predictions.
    st.write("Confusion Matrix: ",conf_matrix)



#functions for prediction models

#function for decisiosn tree prediction
def decision_tree_model(df,x_column, target_column, t_size,max_depth,user_input):  #creating a function with following parametrs
    model = DecisionTreeClassifier(max_depth=max_depth)  #create a instance of the model with the specified number of depts in the tree
    x = df[x_column].values  #selecting the feature column
    y = df[target_column].values #selcting the target column
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=42)  #splitting into train and test
    model.fit(x_train, y_train)  #training the model
    user_input_list = [float(val.strip()) for val in user_input.split(',')]
    # Check if number of user input values matches number of selected columns
    if len(user_input_list) != len(x_column):
        st.error("Number of user input values does not match number of selected columns")
        return
    y_predict = model.predict([user_input_list])
    st.write("The prediction is:", y_predict)

#function for linear regression prediction
def linear_regression_model(df, x_columns, target_column, t_size, user_input): #craete a function with follwoing parameters
    model = LinearRegression() #instance of a model
    X = df[x_columns].values  #selecting  feature column
    y = df[target_column].values  #selecting target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=42) #splitting into train and test
    model.fit(X_train, y_train) #tarining the model
    # Convert user input string to list of values
    user_input_list = [float(val.strip()) for val in user_input.split(',')]
    # Check if number of user input values matches number of selected columns
    if len(user_input_list) != len(x_columns):
        st.error("Number of user input values does not match number of selected columns")
        return
    y_predict = model.predict([user_input_list])
    st.write("The prediction is:", y_predict)

#function for logistic regression prediction
def logistic_regression_model(df, x_column, target_column, t_size, user_input):
    model = LogisticRegression()
    x = df[x_column].values.reshape(-1, 1)  #converts it into a 2d array taht is a single feature with 1 row only
    y = df[target_column].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=42)
    model.fit(x_train, y_train)
    # Convert user input string to list of values
    user_input_list = [float(val.strip()) for val in user_input.split(',')]
    # Check if number of user input values matches number of selected columns
    if len(user_input_list) != 1: #This line checks whether the number of user input values is exactly 1. Logistic regression requires only one input value for prediction.
        st.error("Only one user input value is required for logistic regression")
        return
    # Make prediction based on user input
    user_input_value = float(user_input_list[0])
    y_predict = model.predict([[user_input_value]])
    st.write("The predicted class label is:", y_predict[0])

#function for k means clustering prediction
def k_means_clustering_model(df, x_column, n_clust, user_input):
    x_col = df[[x_column]]
    kmeans = KMeans(n_clusters=n_clust)
    kmeans.fit(x_col)
    cluster = kmeans.predict([[user_input]])
    st.write("Cluster assignment for each data point:")
    st.write(cluster)

#function for knn prediction
def knn_algo_model(df,x_column,y_column,t_size,neighbor_size,user_input):
        x=df[x_column].values
        y=df[y_column].values
        model=KNeighborsClassifier(n_neighbors=neighbor_size)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=t_size,random_state=42)
        model.fit(x_train,y_train)
        # Convert user input string to list of values
        user_input_list = [float(val.strip()) for val in user_input.split(',')]
        # Check if number of user input values matches number of selected columns
        if len(user_input_list) != len(x_column):
            st.error("Number of user input values does not match number of selected columns")
            return
        y_predict = model.predict([user_input_list])
        st.write("The prediction is:", y_predict)

#function for random forest prediction
def random_forest_model(df, x_col, target_column, t_size, user_input):#creating a function with the following parameters as inputs
    if not isinstance(x_col, (list, tuple)): #this checks if the column selected is alist or a tuple
        st.error("x_col must be a list or tuple of column names")  #if not list or tuple return this error
        return
    if not all(col in df.columns for col in x_col): #this line checks if all the column names selected by the user exist in the dataframe or no
        st.error("One or more column names in x_col do not exist in the DataFrame") #if the column does not es=xist in the dataframe than this error is diplayed
        return
    model = RandomForestClassifier()  #create an instance of the model
    x = df[list(x_col)].values  # Select specified columns and converts into 2d array
    y = df[target_column].values  #takes the target column
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=42) #splits the dataset into training and testing set
    model.fit(x_train, y_train) #trains the model based on the above craeted training data
    user_input_list = [float(val.strip()) for val in user_input.split(',')]
    #this takes user input and converst into list and seperated by commas. whitespaces from the floast are removed using strip
    # Check if number of user input values matches number of selected columns
    if len(user_input_list) != len(x_col):
        st.error("Number of user input values does not match number of selected columns")
        return
    # Converts the user input list into a numpy array and reshapes it to have one row and the same number of columns as the features.
    user_input_array = np.array(user_input_list).reshape(1, -1)
    # Make prediction based on user input
    y_predict = model.predict(user_input_array)
    st.write("The prediction is:", y_predict) 





#creating a function to load the first 5 rows of the data so taht we can get an overview of the dataset
if (st.button("Display 5 rows")):
    df=load_data(link) #load the dataset
    st.text("The first 5 rows of the dataset are: ")
    st.write(df.head())  #displaying the first 5 rows on click of the button

st.write("----------------------------------------------------------------------")

status = st.selectbox("Type: ",
                        ['','Data Cleaning and Preprocessing', 'Visualization', 'Model Development through Machine Learning','Prediction using Models'])
#allowig the user to click from the above 3 option to proceed wity the data analysis process
st.write("----------------------------------------------------------------------")
def main():  #craetig the main function 
    if 'Data Cleaning and Preprocessing' in status:  #if the user selected value matches this

            df=load_data(link)  #load the dataset in the begining to start woth the data cleaing and preprocessing process
            #st.form provides a structured and organized way to handle user inputs and actions within the Streamlit app.
            with st.form(key="dimension_form"):
                if st.form_submit_button("Display Dimension"):  #if this button is clicked
                    st.write("The dimension of the dataset are: ")
                    st.write(df.shape)   #print the shape of the dataset

            with st.form(key="columns_form"):
                if st.form_submit_button("Display Columns"): #if this button is clicked
                    st.write("The columns in the dataset are: ")
                    st.write(df.columns)  #dsipaly the columns in the dataset

            with st.form(key="datatypes_form"):
                if st.form_submit_button("Show DataTypes"):  #if this button is clicked
                    st.write("The data types for the columns in the dataset are: ")
                    st.write(df.dtypes)  #display the datatypes of each column

            
            with st.form(key="info_form"):
                if st.form_submit_button("Display Information"):  #if this button is clicked
                    st.write("The information of the dataset is: ")
                    info_s = StringIO()  #this handles strings as file. because the outsput of the df.info is diplayed in the console so to display it in the console we use this object
                    df.info(buf=info_s) #store the df.info in above object
                    st.text(info_s.getvalue())  #dsiplay it in streamlit app


            with st.form(key="unique_val_form"):
                if st.form_submit_button("Check unique values"):  #if this button is clicked
                    val=st.selectbox("Select the column whose unique values are to be checked",options=df.columns)  #sleect from the columns whose unique value is to be hecked
                    if val in df.columns:#if the above selected column is in the dataset
                        unique_values = df[val].unique() #find the unique values and store it in variable
                        num_unique_values = df[val].nunique() #find the count of unique value an dstor it in variable
                        st.write(f"The unique values in column '{val}' are: {unique_values}")  #format the sentence to display answer
                        st.write(f"Total number of unique values: {num_unique_values}")  #format the sentence to display answer

            with st.form(key="value_count_form"):
                if st.form_submit_button("Value Count"):  #if this button is clicked
                    val=st.selectbox("Select the column whose unique values are to be checked",options=df.columns)  #sleect from the columns whose value count is to be checked
                    if val in df.columns:#if the above selected column is in the dataset
                        val_count=df[val].value_counts()
                        st.write(val_count)

            with st.form(key='splitting_column_form'):
                if st.form_submit_button("Split Columns"):  #craeting a button
                    st.write("Delimiter should be same for all the columns to be split")  #information for user
                    cols=st.multiselect("Select the columns to split",df.columns)  #select the column to split
                    delimiter=st.selectbox("Select delimier",options=[',',':',';',' '])  #select teh delimeter
                    split_col(df,cols,delimiter)  #call teh function for splitting clumn that is craeted above           
            

            with st.form(key='renaming_Col_form'):
                if st.form_submit_button("Rename Column"):#craeting a button
                    st.write("The entering of the new name for the column should be in the same order as the column selected to be renamed.")#information for user
                    old_name_col=st.multiselect("Select the columns to rename",df.columns)  #select teh columns whose name is to be replaced with new
                    new_name_col=st.text_input("Enter the new name for the columns")  #enter new column names
                    new_name_col_list = new_name_col.split(",")#split the list by ,
                    col_rename(df,old_name_col,new_name_col_list)  #call the function that is craeted above for renaming the columns


            with st.form(key="na_values_form"):
                if st.form_submit_button("Check NA Values"):  #if this button is clicked
                    st.write("The NA values in the dataset are: ")
                    st.write(df.isna().sum())  #get the count of the na values in each column of the datset

            with st.form(key="duplicate_form"):
                if st.form_submit_button("Check Duplicate Values"):  #if this button is clicked
                    st.write("The duplicate values in the dataset are: ")
                    st.write(df.duplicated().sum())  #this checks the duplicated values in each column in the datset

            with st.form(key="dropna_form"):
                if st.form_submit_button("Drop NA Values"):  #if this button is clicked
                    st.text("Dimensions before dropping NA")
                    st.write(df.shape)  #get the shape of the dataset before dropping the na values
                    df_new = df.dropna()
                    st.text("Dimensions after dropping NA")
                    st.write(df_new.shape)  #displayi g the shape of the datset after dropping the na values to see the change
                    st.write(df_new) #print the new datset. you can also download this new dataset

        
            with st.form(key='column_drop_form'):
                if st.form_submit_button("Drop Column"):  #if this button is clicked
                    column_name = st.multiselect("Select Column to change", df.columns)  #sleect the columns which are to be dropped
                    drop_col(df,column_name)  #drop the columns. this is the calling of the function craeted above
            


            with st.form(key="replace_na_mean"):
                if st.form_submit_button("Replace NA with mean"): #if this button is clicked
                    cols=st.multiselect("Select columns whose value is to be replaced",df.columns)  #select teh columns whose na values are to be replaced by th mean
                    replace_mean(df,cols) #calling of the function craeted above

            with st.form(key="replace_na_median"):
                if st.form_submit_button("Replace NA with median"): #if this button is clicked
                    cols=st.multiselect("Select columns whose value is to be replaced",df.columns) #sleect the colums whose na value sare to be replaced by the median
                    replace_median(df,cols)  #calling the function craeted above to replace with median
            

            with st.form(key="replace_na_default"):
                if st.form_submit_button("Replace NA with Default"):#if this button is clicked
                    val=st.text_input("Select the default value") #take the input value from the user which is to be replaced by na
                    cols=st.multiselect("Select columns whose value is to be replaced",df.columns)  #column whose na values are to be replaced by the user defined value
                    replace_default(df,cols,val)  #calling the above craeted function

            with st.form(key="uniqueness_form"):
                if st.form_submit_button("Filter Records"): #if this button is clicked
                    st.write("This will filter as per the Categorical column")  #print this
                    col=st.selectbox("Select the column on which to apply the filter",df.columns) #sleect the column based on whom filteration is to be done
                    val=st.multiselect("Selection of filtering Value",df[col].unique()) #give user the opyions to selct from unique values of that filtering column
                    filter(df,col,val) #call the function craeted above

            with st.form(key="greater_form"):
                if st.form_submit_button("Filter Record by Greater Than Value"): #if this button is clicked
                    st.write("This will filter as per the Numerical column")  #print this
                    col=st.selectbox("Select the column on which to apply the filter",df.columns)  #Selct the colum o the basis on which filtering is to done
                    val=st.number_input("Enter the threshold Value")  #select thresholding value
                    greater_record(df,col,val) #call the function that is created above


            with st.form(key="lesser_form"):
                if st.form_submit_button("Filter Record by Lesser Than Value"):#if this button is clicked
                    st.write("This will filter as per the Numerical column") #print this
                    col=st.selectbox("Select the column on which to apply the filter",df.columns) #Selct the colum o the basis on which filtering is to done
                    val=st.number_input("Enter the threshold Value")  #select thresholding value
                    lessthan_record(df,col,val) #call the function that is created above





    elif 'Visualization' in status:  #if the option slected by the user is visualization
        df=load_data(link)  #load the data to use for visualization
        
        #creating a button for barplot
        #st.form provides a structured and organized way to handle user inputs and actions within the Streamlit app.
        with st.form(key='barplot_form'):
            if st.form_submit_button("Create Barplot"):  #if this button is clicked
                st.write("A bar plot is constructed by utilizing two variables, where one variable is designated for the categorical representation along the X-axis, and the other variable is assigned to the numerical values depicted along the Y-axis.")
                x_col= st.selectbox("Select X Column", df.columns) #select the x axis column for the chart
                y_col= st.selectbox("Select Y Column", df.columns) #select the y axis column for the chart
                x_Axis=st.text_input("Enter title for x axis") #enter the title for the x axis
                y_Axis=st.text_input("Enter title for y axis") #enter the title for the y axis
                plot_title = st.text_input("Enter Plot Title", "Bar Plot")   #enter the title for the plot. default is bar plot
                plot = create_barplot(df, x_col, y_col,x_Axis,y_Axis, plot_title) #call the function craeted above  for bar plot and store it in the variable
                st.plotly_chart(plot) #call that variable using the plotly chart function

    

        #creating a button for sns count plot
        with st.form(key='sns_countplot_form'):  
            if st.form_submit_button("Create Countplot"):
                st.write("A countplot is configured with a singular variable exclusively assigned to the X-axis, while the Y-axis denotes the frequency count of the occurrences associated with the specified variable along the X-axis.")
                x_col= st.selectbox("Select X Column", df.columns)  #select the x axis column for the chart            
                x_Axis=st.text_input("Enter title for x axis") #enter the title for the x axis
                y_Axis=st.text_input("Enter title for y axis","count")#enter the title for the y axis.default is count
                plot_title = st.text_input("Enter Plot Title", "CountPlot")  #enter the title for the plot. default is countplot
                plot = create_snsplot(df, x_col,x_Axis,y_Axis, plot_title)    #call the function craeted above  for count plot and store it in the variable                     
                st.pyplot(plot)  #this pyplot is sue to display the seaborn countplot
                tmpfile_path = save_plot_as_jpeg(plot, 'countplot download')
                st.markdown(get_download_link(tmpfile_path, 'countplot download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)
            
        

        #createing a button for scatter plot
        with st.form(key='scatter_plot_form'):
            if st.form_submit_button("Create Scatter Plot"): #if this button is clicked
                st.write("A scatter plot is constructed by employing two numerical variables, with one designated for the X-axis and the other for the Y-axis. The representation of data points on the plot illustrates the relationship between these two numerical variables.")
                x_col= st.selectbox("Select X Column", df.columns) #select the x axis column for the chart  
                y_col= st.selectbox("Select Y Column", df.columns)  #select the y axis column for the chart         
                x_Axis=st.text_input("Enter title for x axis")  #enter the title for the x axis
                y_Axis=st.text_input("Enter title for y axis")  #enter the title for the y axis
                plot_title = st.text_input("Enter Plot Title", "Scatter Plot")  #enter the title for the plot. default is scatter plot
                plot = create_scatterplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)      #call the function craeted above  for scatter plot and store it in the variable     
                st.plotly_chart(plot) #dsiplay the above careted variable suing the plotly chart method
                


            

        #creating a button for line plot
        with st.form(key='line_plot_form'):
            if st.form_submit_button("Create Line Plot"): #if this button is clicked
                st.write("A line plot is generated by associating a date column with the X-axis variable and another variable with the Y-axis. This visualization method allows for the depiction of trends and patterns over time.")
                x_col= st.selectbox("Select X Column", df.columns) #select the x axis column for the chart
                y_col= st.selectbox("Select Y Column", df.columns)   #select the y axis column for the chart            
                x_Axis=st.text_input("Enter title for x axis") #enter the title for the x axis
                y_Axis=st.text_input("Enter title for y axis") #enter the title for the y axis
                plot_title = st.text_input("Enter Plot Title", "Line Plot")  #enter the title for the plot. default is line plot
                plot = create_lineplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)         #call the function craeted above  for line plot and store it in the variable               
                st.plotly_chart(plot)   #dsiplay the above careted variable suing the plotly chart method  


        #creating a button for histogram
        with st.form(key="histogram_form"):
            if st.form_submit_button("Create Histogram"): #if this button is clicked
                st.write("A histogram plot is constructed by utilizing a single variable assigned to the X-axis, typically representing the distribution of numerical data, while the Y-axis remains dedicated to denoting the frequency or density of observations within specified intervals along the X-axis.")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                x_Axis=st.text_input("Enter title for X axis")  #enter the title for the x axis
                y_Axis=st.text_input("Enter title for Y Axis")  #enter the title for the y axis
                plot_title=st.text_input("Enter title for the plot","Histogram")  #enter the title for the plot. default is histogram
                plot=create_histogram(df,x_col,y_col,x_Axis,y_Axis,plot_title)  #call the function craeted above  for histogram and store it in the variable 
                st.plotly_chart(plot)  #dsiplay the above careted variable suing the plotly chart method 


        #creating a button for bubble plot  
        with st.form(key="bubbleplot_form"):
            if st.form_submit_button("Create Bubble Plot"): #if this button is clicked
                st.write("A bubble plot employs three variables, designating one to the X-axis, another to the Y-axis, and a third for the size of the bubbles. Furthermore, a fourth parameter is utilized to introduce color.")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                size_1=st.selectbox("Select column for Size",df.columns)  #sleect the column for specifying the size of the bubles
                col=st.selectbox("Select column for color",df.columns)  #select the column for classifyig the points into different classes with different colors
                x_Axis=st.text_input("Enter title for x axis")  #enter the title for the x axis
                y_Axis=st.text_input("Enter title for Y axis")  #enter the title for the y axis
                plot_title=st.text_input("Enter title for the plot","Bubble Plot")  #enter the title for the plot. default is bubble plot
                plot=create_bubbleplot(df,x_col,y_col,size_1,col,x_Axis,y_Axis,plot_title)  #call the function craeted above  for bubble plot and store it in the variable 
                st.plotly_chart(plot)  #dsiplay the above careted variable suing the plotly chart method 


        #creating a button for boxplot
        with st.form(key="boxplot_form"):
            if st.form_submit_button("Create Boxplot"):  #if this button is clicked
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                x_Axis=st.text_input("Enter title for X axis")   #enter the title for the x axis
                y_Axis=st.text_input("Enter title for Y Axis")  #enter the title for the y axis
                plot_title=st.text_input("Enter title for the plot","Box Plot")  #enter the title for the plot. default is box plot
                plot=box_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)  #call the function craeted above  for box plot and store it in the variable 
                st.plotly_chart(plot)  #dsiplay the above careted variable suing the plotly chart method 

        with st.form(key="pie_chart"):
            if st.form_submit_button("Create Pie Chart"):
                val=st.selectbox("Select the value for the piechart slices",df.columns)
                names=st.selectbox("Select the names for the slices",df.columns)
                title=st.text_input("Select the title for the chart","Pie Chart")
                plot=create_piechart(df,val,names,title)
                st.plotly_chart(plot)

        #creating a button for voilin chart
        with st.form(key="violinchart_form"):
            if st.form_submit_button("Create Violin Chart"):  #if this button is clicked
                st.write("A violin chart is constructed by employing two variables, where one is designated for the X-axis and the other for the Y-axis. This visualization method effectively portrays the distribution and density of the Y-axis variable across different levels or categories specified along the X-axis.")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                x_Axis=st.text_input("Enter title for X axis")   #enter the title for the x axis
                y_Axis=st.text_input("Enter title for Y Axis")   #enter the title for the y axis
                plot_title=st.text_input("Enter title for the plot","Voilin Plot")  #enter the title for the plot. default is  voilin plot
                plot=create_violin_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)  #call the function craeted above  for  voilin plot and store it in the variable 
                st.plotly_chart(plot)   #dsiplay the above careted variable suing the plotly chart method 


        #creating a button for gantt chart
        with st.form(key="gantt_chart_form"):
            if st.form_submit_button("Create Gantt Chart"):  #if this button is clicked
                st.write("A Gantt chart utilizes temporal variables for the X-axis, with the X start and X end columns representing the project's initiation and completion dates. The Y-axis accommodates a categorical variable, offering a concise depiction of project components and timelines.e")
                x_start_column=st.selectbox("Select X Start Column",df.columns)  #select the  column for starting of x axis in the chart
                x_end_column=st.selectbox("Select X End Column",df.columns)  #select the  column for ending of x axis in the chart
                y_column_name=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                x_Axis=st.text_input("Enter title for X axis")  #enter the title for the x axis
                y_Axis=st.text_input("Enter title for Y Axis")  #enter the title for the y axis
                plot_title=st.text_input("Enter title for the plot","Gantt Plot")   #enter the title for the plot. default is  gantt plot
                plot=gantt_chart(df,x_start_column,x_end_column,y_column_name,x_Axis,y_Axis,plot_title)  #call the function craeted above  for gantt plot and store it in the variable
                st.plotly_chart(plot)  #dsiplay the above careted variable suing the plotly chart method 



        #creating a button for wordcloud
        with st.form(key="wordcloud_form"):
            if st.form_submit_button("Create Wordcloud"):  #if this button is clicked
                st.write("A wordcloud is used to find the most frequently occuring words in the reviews or in the text. Select the column that contains text or reviews.")
                review_col=st.selectbox("Select the review column",df.columns)  #select the column of text on the basis of which wordcloud will be created
                color=st.selectbox("Select the background color",['black','white'])  #select the baclground color for the wordcloud
                title=st.text_input("Select the title for wordcloud")  #enter the title for the wordcloud
                plot=create_wordcloud(df,review_col,color,title)   #call the function craeted above  for wordcloud and store it in the variable
                st.pyplot(plot) #this pyplot is sue to display the seaborn plot
                tmpfile_path = save_plot_as_jpeg(plot, 'wordcloud download')
                st.markdown(get_download_link(tmpfile_path, 'wordcloud download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)


        #creating a button for 3D line plot
        with st.form(key="3D_lineplot_form"):
            if st.form_submit_button("Create 3D Lineplot"):  #if this button is clicked
                st.write("A 3D line plot visualizes data in three dimensions, typically using X variable, Y variable, and Z variable  to represent the relationship between variables and display trends or patterns in a three-dimensional space.")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                z_col=st.selectbox("Select Z Column",df.columns)  #select the z axis column for the chart
                plot_title=st.text_input("Enter title for the plot","3D Line Plot")  #enter the title for the plot. default is 3D line plot
                plot=lineplot_3d(df,x_col,y_col,z_col,plot_title)  #call the function craeted above  for 3d lineplot and store it in the variable
                st.plotly_chart(plot) #dsiplay the above careted variable suing the plotly chart method 
                


        #creating  a button for 3D scatter plot
        with st.form(key="3D_scatterplot_form"):
            if st.form_submit_button("Create 3D ScatterPlot"):  #if this button is clicked
                st.write("A 3D scatter plot represents data points in a three-dimensional space, using three numerical variables (X, Y, Z) to visualize the distribution and relationships among data points in a more comprehensive manner")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                z_col=st.selectbox("Select Z Column",df.columns)  #select the z axis column for the chart
                col=st.selectbox("Select Column for color",df.columns) #select the column to differentiate the data points into different categories of a column based on different colors
                plot_title=st.text_input("Enter title for the plot","3D Scatter Plot")  #enter the title for the plot. default is 3D scatter plot
                plot=scatterplot_3d(df,x_col,y_col,z_col,col,plot_title)  #call the function craeted above  for 3d scatterplot and store it in the variable
                st.plotly_chart(plot)  #dsiplay the above careted variable suing the plotly chart method 


        #creating a button for distplot
        with st.form(key="distplot_form"):
            if st.form_submit_button("Create Distplot"):  #if this button is clicked
                st.write("A distplot offers a graphical depiction of the univariate distribution of a dataset. Specifically designed for single-variable analysis, it elegantly combines a histogram and a kernel density estimate.")
                x_col=st.selectbox("Select X Column",df.columns)   #select the x axis column for the chart
                plot_title=st.text_input("Enter title for the plot","DistPlot")  #enter the title for the plot. default is distplot
                plot=dist_plot(df,x_col,plot_title)  #call the function craeted above  for distplot and store it in the variable
                st.pyplot(plot)  #this pyplot is sue to display the seaborn plot
                tmpfile_path = save_plot_as_jpeg(plot, 'distplot download')
                st.markdown(get_download_link(tmpfile_path, 'distplot download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)


        #creating a button for jointplot
        with st.form(key="jointplot_form"):
            if st.form_submit_button("Create Jointplot"):  #if this button is clicked
                st.write("A joint plot visually combines two bivariate plots, such as scatter plots and histograms, to illustrate the relationship between two variables in a single comprehensive representation.")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                plot_title=st.text_input("Enter title for the plot","Joint Plot")  #enter the title for the plot. default is joint plot
                plot=joint_plot(df,x_col,y_col,plot_title)  #call the function craeted above  for jointplot and store it in the variable
                st.pyplot(plot) #this pyplot is sue to display the seaborn plot
                tmpfile_path = save_plot_as_jpeg(plot, 'jointplot download')
                st.markdown(get_download_link(tmpfile_path, 'jointplot download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)
                
        #craeting a button for pairplot
        with st.form(key="pairplot_form"):
            if st.form_submit_button("Create Pairplot"): #if this button is clicked
                st.write("A pairplot is a matrix of scatterplots and histograms that provides a quick and comprehensive visual exploration of pairwise relationships within a multivariate dataset.")
                plot_title=st.text_input("Enter title for the plot","Pair Plot")  #enter the title for the plot. default is pair plot
                plot=pair_plot(df,plot_title)  #call the function craeted above  for pairplot and store it in the variable
                st.pyplot(plot)   #this pyplot is sue to display the seaborn plot
                tmpfile_path = save_plot_as_jpeg(plot, 'pairplot download')
                st.markdown(get_download_link(tmpfile_path, 'pairplot download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)


        #creating a button for kde plot
        with st.form(key="kdeplot_form"):
            if st.form_submit_button("Create KDEplot"):  #if this button is clicked
                st.write("A KDE (Kernel Density Estimation) plot offers a smoothed representation of the probability density function for a continuous variable, facilitating a visual estimate of the underlying distribution. Typically it is constructed with two variables, one allocated to the X-axis and the other to the Y-axis")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                plot_title=st.text_input("Enter title for the plot","KDE Plot")  #enter the title for the plot. default is kde plot
                plot=kde_plot(df,x_col,y_col,plot_title) #call the function craeted above  for kde plot and store it in the variable
                st.pyplot(plot)   #this pyplot is sue to display the seaborn plot
                tmpfile_path = save_plot_as_jpeg(plot, 'kde plot download')
                st.markdown(get_download_link(tmpfile_path, 'kde plot download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)


        #creating a button for Regression plot
        with st.form(key="regplot_form"):
            if st.form_submit_button("Create Regplot"):  #if this button is clicked
                st.write("A regplot visually depicts the relationship between two continuous variables through a scatter plot with a fitted regression line, aiding in the exploration of potential linear associations and trends in the data.")
                x_col=st.selectbox("Select X Column",df.columns)  #select the x axis column for the chart
                y_col=st.selectbox("Select Y Column",df.columns)  #select the y axis column for the chart
                plot_title=st.text_input("Enter title for the plot","Regression Plot")  #enter the title for the plot. default is regression plot 
                plot=reg_plot(df,x_col,y_col,plot_title)  #call the function craeted above  for reg plot and store it in the variable
                st.pyplot(plot)  #this pyplot is sue to display the seaborn plot
                tmpfile_path = save_plot_as_jpeg(plot, 'regression plot download')
                st.markdown(get_download_link(tmpfile_path, 'regression plot download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)

        #creating a button for Heatmap
        with st.form(key="heatmap_form"):
            if st.form_submit_button("Create Heatmap"): #if this button is clicked
                st.write("A heatmap is a visual representation of data in a matrix format, using color gradients to convey variations in values and facilitating the identification of patterns, correlations, or intensity within the dataset.")
                plot=create_heatmap(df)#call the function craeted above  for heatmap and store it in the variable
                st.pyplot(plot) #this pyplot is sue to display the seaborn plot
                tmpfile_path = save_plot_as_jpeg(plot, 'heatmap download')
                st.markdown(get_download_link(tmpfile_path, 'heatmap download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)

        #creating a button for stacked plot
        with st.form(key="stacked_barplot_form"):
            if st.form_submit_button("Create Stacked Plot"): #if this button is clicked
                st.write("A stacked bar chart provides a visual representation of categorical data incorporating multiple variables, where the X-axis denotes the categorical dimension, and the stacked bars on the Y-axis illustrate the cumulative and segmented contributions of two additional variables")
                x_col=st.selectbox("Select X Column",df.columns) #select the x axis column for the chart
                y1_col=st.selectbox("Select first comparison Column",df.columns)  #select the first y axis column for the chart
                y2_col=st.selectbox("Select second comparison Column",df.columns)  #select the second y axis column for the chart
                plot_title=st.text_input("Enter title for the plot","Stacked Plot")  #enter teh title for the chart. default value is stacked plot
                plot=stacked_bar(df,x_col,y1_col,y2_col,plot_title)#call the function craeted above  for stacked barplot and store it in the variable
                st.plotly_chart(plot)  #this pyplot is sue to display the seaborn plot
                

        #craeting a button double line plot
        with st.form(key="double_lineplot_form"):
            if st.form_submit_button("Create Double LinePlot"):  #if this button is clicked
                st.write("A double line plot provides a graphical representation of two distinct variables over a common axis, with the X-axis designating the categorical or temporal dimension and the paired Y-axes depicting the trends of the two variables.")
                x_col=st.selectbox("Select X Column",df.columns) #select the x axis column for the chart
                y1_col=st.selectbox("Select first comparison Column",df.columns) #select the first y axis column for the chart
                y2_col=st.selectbox("Select second comparison Column",df.columns) #select the second y axis column for the chart
                plot_title=st.text_input("Enter title for the plot","Line Plot") #enter teh title for the chart. default value is line plot
                plot=double_line(df,x_col,y1_col,y2_col,plot_title) #call the function craeted above  for double lineplot and store it in the variable
                st.plotly_chart(plot)  #this pyplot is sue to display the seaborn plot

        #creating a button for candlestick chart
        with st.form(key="candlestick_chart_form"):
            if st.form_submit_button("Create Candlestick Chart"): #if this button is clicked
                st.write("A candlestick chart serves as a visual representation of financial price dynamics, employing a date variable on the X-axis and incorporating four essential parameters – open, close, high, and low – as distinct elements within candlestick shapes. This methodical presentation facilitates meticulous trend analysis and pattern recognition in financial markets.")
                x_col=st.selectbox("Select X Column",df.columns)  #select the column for date on the x axis.basically the timefarme column
                open_col=st.selectbox("Select open prices Column",df.columns)  #select the column for opening price
                high_col=st.selectbox("Select high prices Column",df.columns)  #select the column for high price
                low_col=st.selectbox("Select low prices Column",df.columns)  #select the column for low price
                close_col=st.selectbox("Select close prices Column",df.columns)  #select the column for closing price
                plot=candlestick_chart(df,x_col,open_col,high_col,low_col,close_col)  #call the function craeted above  for candlestick plot and store it in the variable
                st.plotly_chart(plot)  #use the plotly chart method to display the chart

        #button for sentiment analysis
        with st.form(key="sentiment_form"):
            if st.form_submit_button("Sentiment analysis"):  #if this button is clicked
                st.write("Sentiment analysis with polarity scoring assigns numerical values to text data, indicating whether the sentiment expressed is positive, negative, or neutral, enabling efficient extraction of sentiment-related insights.")
                col=st.selectbox("Select the reviews column",df.columns)  #select the column whose analysis is to be done
                df_display, sentiment_counts,plot = sentiment_analysis(df, col)  #call the above function for snetiment analysis
                st.write(df_display)  #display the dataset
                st.write("Sentiment Counts:") #print this
                st.write(sentiment_counts) #display the sentiment count
                st.pyplot(plot) #display the plot
                tmpfile_path = save_plot_as_jpeg(plot, 'sentiment analysis chart download')
                st.markdown(get_download_link(tmpfile_path, 'sentiment analysis chart download'), unsafe_allow_html=True)
                os.remove(tmpfile_path)


    elif "Model Development through Machine Learning" in status:  # if teh option selected by user is this
        #creating a button for simple linear Regression
        df=load_data(link)  #load the dataset

        with st.form(key="linear_regression_form"):
            if st.form_submit_button("Linear Regression"):  #if this button is clicked
                x_cols = st.multiselect("Select the independent columns", df.columns)  #select the independent features's columns
                target_column = st.selectbox("Select the Target column", df.columns)  #select the target variable column
                t_size = st.number_input("Enter test size")  #enter the test size i.e the size of the testing part of the data
                linear_regression(df, x_cols, target_column, t_size) #call the function craeted above for linear regression


        with st.form(key="logistic_regression_form"):
            if st.form_submit_button("Logistic Regression"): #if this button is clicked
                x_col = st.multiselect("Select the independent column", df.columns)  #select the independent features's columns
                target_column = st.selectbox("Select the Target column", df.columns)  #select the target variable column
                t_size = st.number_input("Enter test size") #enter the test size i.e the size of the testing part of the data
                logistic_regression(df, x_col, target_column, t_size) #call the function craeted above for logistic regression


        with st.form(key="knn_form"):
            if st.form_submit_button("K Nearest Neighbor "):  #if this button is clicked
                x_col = st.multiselect("Select the independent column", df.columns) #select the independent features's columns
                target_column = st.selectbox("Select the Target column", df.columns) #select the target variable column
                t_size = st.number_input("Enter test size")  #enter the test size i.e the size of the testing part of the data
                neighbor_size=st.number_input("Enter the neighbors value",min_value=2,max_value=10)  #select the number of the enighbors to be craeted
                knn_algo(df,x_col,target_column,t_size,neighbor_size) #call the function craeted above for knn algorithmn


        with st.form(key="decision_tree_form"):
            if st.form_submit_button("Decision Tree Classifier "):  #if this button is clicked
                x_col = st.multiselect("Select the independent column", df.columns)  #select the independent features's columns
                target_column = st.selectbox("Select the Target column", df.columns) #select the target variable column
                t_size = st.number_input("Enter test size")   #enter the test size i.e the size of the testing part of the data
                max_depth=st.number_input("Enter the depth value",min_value=2,max_value=10)  #enter the levels in the decision tree
                decision_tree(df,x_col, target_column, t_size,max_depth) #call the function craeted above for decision tree

        with st.form(key="decision_tree_category_form"):
            if st.form_submit_button("Decision Tree Classifier for categorical"):  #if this button is clicked
                x_col = st.multiselect("Select the independent column", df.columns)  #select the independent features's columns
                target_column = st.selectbox("Select the Target column", df.columns) #select the target variable column
                t_size = st.number_input("Enter test size")   #enter the test size i.e the size of the testing part of the data
                max_depth=st.number_input("Enter the depth value",min_value=2,max_value=10)  #enter the levels in the decision tree
                decision_tree_for_cat(df,x_col, target_column, t_size,max_depth) #call the function craeted above for decision tree

        with st.form(key="clustering_algorithmn_form"):
            if st.form_submit_button("Clustering Algorithmn "): #if this button is clicked
                x_col = st.selectbox("Select the features column", df.columns) #select the independent features's columns
                n_clust=st.number_input("Enter the clusters number",min_value=2,max_value=10)  #enter the  number of the clusters
                k_means_clustering(df,x_col,n_clust) #call the function craeted above for clustering algorithmn
        
        with st.form(key="random_forest_form"):
            if st.form_submit_button("Random Forest "): #if this button is clicked
                x_col = st.multiselect("Select the independent column", df.columns) #select the independent features's columns
                target_column = st.selectbox("Select the Target column", df.columns) #select the target variable column
                t_size = st.number_input("Enter test size") #enter the test size i.e the size of the testing part of the data
                random_forest(df,x_col,target_column,t_size) #call the function craeted above for random forest


    elif 'Prediction using Models' in status:
        df=load_data(link)
        #button for linear regression
        with st.form(key="linear_regression_pred_form"):
            if st.form_submit_button("Linear Regression"):
                x_cols = st.multiselect("Select the independent columns", df.columns)
                target_column = st.selectbox("Select the Target column", df.columns)
                t_size = st.number_input("Enter test size")
                user_input = st.text_input("Enter the values to predict (separated by commas)")
                linear_regression_model(df, x_cols, target_column, t_size, user_input)

        #button for logistic regression
        with st.form(key="logistic_regression_pred_form"):
            if st.form_submit_button("Logistic Regression"):
                x_col = st.selectbox("Select the independent column", df.columns)
                target_column = st.selectbox("Select the Target column", df.columns)
                t_size = st.number_input("Enter test size")
                user_input = st.text_input("Enter the value to predict")
                logistic_regression_model(df, x_col, target_column, t_size, user_input)

        #button for clustering
        with st.form(key="clustering_algorithm_pred_form"):
            if st.form_submit_button("Clustering Algorithm"):
                x_col = st.selectbox("Select the feature column", df.columns)
                n_clust = st.number_input("Enter the number of clusters", min_value=2, max_value=10)
                user_input = st.number_input("Enter the value")
                k_means_clustering_model(df, x_col, n_clust, user_input)

        #button for knn
        with st.form(key="knn_pred_form"):
            if st.form_submit_button("K Nearest Neighbor "):
                x_col = st.multiselect("Select the independent column", df.columns)
                target_column = st.selectbox("Select the Target column", df.columns)
                t_size = st.number_input("Enter test size")
                neighbor_size=st.number_input("Enter the neighbors value",min_value=2,max_value=10)
                user_input=st.text_input("Enter values")
                knn_algo_model(df,x_col,target_column,t_size,neighbor_size,user_input)


        #button fr decision tree
        with st.form(key="decision_tree_pred_form"):
            if st.form_submit_button("Decision Tree Classifier "):
                x_col = st.multiselect("Select the independent column", df.columns)
                target_column = st.selectbox("Select the Target column", df.columns)
                t_size = st.number_input("Enter test size")
                max_depth=st.number_input("Enter the depth value",min_value=2,max_value=10)
                user_input = st.text_input("Enter the value to predict")
                decision_tree_model(df,x_col, target_column, t_size,max_depth,user_input)

        #button for random forest
        with st.form(key="random_forest_pred_form"):
            if st.form_submit_button("Random Forest "):
                x_col = st.multiselect("Select the independent column", df.columns)
                target_column = st.selectbox("Select the Target column", df.columns)
                t_size = st.number_input("Enter test size")
                user_input = st.text_input("Enter the value to predict")
                random_forest_model(df,x_col,target_column,t_size,user_input)

main()  #calling the main function so that the app runs

