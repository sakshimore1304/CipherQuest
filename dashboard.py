import streamlit as st   #library for web application craetion
import pandas as pd     #for data manipulation
import numpy as np    #for numerical computation
import matplotlib.pyplot as plt    #for creating visualizatuons
import seaborn as sns     #for creating statistical and complex visualization
import plotly.express as px   #generating complex visualizations with minimal code
import plotly.graph_objects as go  #provides a flexible and powerful interface for constructing various types of plots and figures. 
from wordcloud import WordCloud  #to create wordcloud
from textblob import TextBlob #for text data analysis
import base64  #for encoding

st.set_page_config(layout="wide")

def get_base64_of_bin_file(bin_file): # a function that takes input as bin_file which is the finary file
    with open(bin_file, 'rb') as f: #opens binary file in te binary mode. with ensures closing teh file after operations
        data = f.read() #reads the content of the file and stores it in the data variable
    return base64.b64encode(data).decode()  #converts the binary data into a Base64 encoded byte string.It decodes the Base64 encoded byte string into a UTF-8 string.decode is necesssary because b64 returns byts and we want string

def set_png_as_page_bg(png_file): #defines a function that takes png file as input and display it in the background of the webpage
    bin_str = get_base64_of_bin_file(png_file)  #calls the function get_base64_of_bin_file(png_file) to convert the PNG image file specified by png_file into its Base64 encoded representation.
    page_bg_img = ''' 
    <style> 
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Use this function to set the background image of your app
set_png_as_page_bg('bg2.png')




#giving the title to the sidebar
st.sidebar.title("CipherQuest")
st.sidebar.title("Welcome!")

#taking the title from the user for the Dashboard
title=st.sidebar.text_input("Enter the title for Dashboard!")
st.title(title)
subtitle=st.sidebar.text_input("Enter subheader")
st.subheader(subtitle)
cap=st.sidebar.text_input("Enter the caption")
st.caption(cap)
#option to upload the file for creating the dataset
link=st.sidebar.file_uploader("Select the file", type=["xlsx", "csv"])

#function for opening the file
def open_data(file):
    if file is not None:
        if file.name.endswith(".xlsx"):  #checks if the file ends with xlsx i.e whether it is a excel file
            data=pd.read_excel(file) #if yes it loads the file
        elif file.name.endswith(".csv"): #if the file does not end with xlsx it ends with csv 
            data=pd.read_csv(file)#load that csv file
        return data  #return that dataset for futher processing

#loading the uploaded dataset
df=open_data(link)  #load the datset by callig the function

#barplot
def create_barplot(df, x_column, y_column,x_axis,y_axis ,title): #creating a function with the desired parameters
    fig = px.bar(df, x=x_column, y=y_column, title=title)  #create a bar chart using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis) #customizing the chart with x and y axis title
    return fig #returing the figure to diplay it

#countplot
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

#scatter plot
def create_scatterplot(df,x_column,y_column,x_axis,y_axis,title):  #creating a function with the desired parameters
    fig=px.scatter(df,x=x_column,y=y_column,title=title)  #create a scatter chart using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis) #customizing the chart with x and y axis title
    return fig #returing the figure to diplay it

#line plot
def create_lineplot(df,x_column,y_column,x_axis,y_axis,title): #creating a function with the desired parameters
    fig=px.line(df,x=x_column,y=y_column,title=title)#create a line plot using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis) #customizing the chart with x and y axis title
    return fig #returing the figure to diplay it



#creating a function for histogram
def create_histogram(df,x_column,y_column,x_axis,y_axis,title): #creating a function with the desired parameters
    fig=px.histogram(df,x=x_column,y=y_column,title=title) #create a histogram using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis)  #customizing the chart with x and y axis title
    return fig #returing the figure to diplay it


#creating a function for bubble chart
def create_bubbleplot(df,x_column,y_column,size_1,color_1,x_axis,y_axis,title): #creating a function with the desired parameters
    fig=px.scatter(df,x=x_column,y=y_column,size=size_1,title=title,color=color_1) #create a bubble plot using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis) #customizing the chart with x and y axis title
    return fig  #returing the figure to diplay it


#creating a function for voilin chart
def create_violin_plot(df,x_column,y_column,x_axis,y_axis,title): #creating a function with the desired parameters
    fig=px.violin(df,x=x_column,y=y_column,title=title) #create a voilin chart using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis) #customizing the chart with x and y axis title
    return fig #returing the figure to diplay it


#creating a function for gantt chart
def gantt_chart(df,start,finish,y_column,x_axis,y_axis,title): #creating a function with the desired parameters
    fig=px.timeline(df,x_start=start,x_end=finish,y=y_column,title=title) #create a gantt chart using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis) #customizing the chart with x and y axis title
    return fig #returing the figure to diplay it


#creating a function for boxplot
def box_plot(df,x_column,y_column,x_axis,y_axis,title): #creating a function with the desired parameters
    fig=px.box(df,x=x_column,y=y_column,title=title) #create a box plot using plotly function and the follwoing parameters
    fig.update_layout(xaxis_title=x_axis,yaxis_title=y_axis) #customizing the chart with x and y axis title
    return fig #returing the figure to diplay it


#creating a function for wordcloud
def create_wordcloud(df,x_column,col,title): #creating a function with the desired parameters
    column=' '.join(df[x_column].astype(str)) #taking the text column and joining all rows into a single string seperating it with space and store in a new variable
    wordcloud=WordCloud(width=800,height=800,background_color=col).generate(column) #craete a wordcloud and generate using the above craeted new variable
    plt.figure(figsize=(10, 10))  #adjust the size of the chart
    plt.imshow(wordcloud, interpolation='bilinear') #displaying the wordcloud
    plt.title(title)  #displaying the title for the chart
    plt.axis('off')  # Turn off axis
    return plt #returing the figure to diplay it


#creating a function for 3d  line plot
def lineplot_3d(df,x_column,y_column,z_column,title): #creating a function with the desired parameters
    fig=px.line_3d(df,x=x_column,y=y_column,z=z_column,title=title) #creating a 3d line plot using the plotly function and with the desired parameters
    return fig #returing the figure to diplay it


#creating a function for 3d scatter plot
def scatterplot_3d(df,x_column,y_column,z_column,color,title): #creating a function with the desired parameters
    fig=px.scatter_3d(df,x=x_column,y=y_column,z=z_column,color=color,title=title) #creating a 3d scatter plot using the plotly function and with the desired parameters
    return fig #returing the figure to diplay it


#creating a function for distplot
def dist_plot(df,x_column,title): #creating a function with the desired parameters
    fig=sns.displot(df[x_column],kde=True) #craeting a displot with the seaborn library and the follwoing parameters
    plt.title(title) #giving user defined title to the chart
    return fig #returing the figure to diplay it


#creating a function for jointplot
def joint_plot(df,x_column,y_column,title): #creating a function with the desired parameters
    fig=sns.jointplot(df,x=x_column,y=y_column) #craeting a jointplot with the seaborn library and the follwoing parameters
    plt.title(title)  #giving user defined title to the chart
    plt.xticks(rotation=45)
    return fig #returing the figure to diplay it



#creating a function for kde plot
def kde_plot(df,x_column,y_column,title):  #creating a function with the desired parameters
    fig=sns.kdeplot(df,x=x_column,y=y_column) #craeting a kdeplot with the seaborn library and the follwoing parameters
    plt.title(title) #giving user defined title to the chart
    return fig #returing the figure to diplay it


#creating a function for regression plot
def reg_plot(df,x_column,y_column,title):  #creating a function with the desired parameters
    fig=sns.lmplot(df,x=x_column,y=y_column) #craeting a regression plot with the seaborn library and the follwoing parameters
    plt.title(title) #giving user defined title to the chart
    return fig #returing the figure to diplay it

#creating a function for heatmap
def create_heatmap(df):
    df_corr = df.corr()
    fig = plt.figure(figsize=(10, 8))  # Create a new figure
    sns.heatmap(df_corr, annot=True, cmap='viridis')  # Plot the heatmap with annotations and colormap
    plt.title('Correlation Heatmap')  # Set the title of the heatmap
    plt.savefig('my_plot.png')  # Save the figure as an image
    plt.close(fig)  # Close the figure to release memory
    return fig

#creating a function for comparing 2 bars in a plot(stacked bar chart)
def stacked_bar(df,x_column,y1_column,y2_column,title): #creating a function with the desired parameters
    fig=px.bar(df,x=x_column,y=[y1_column,y2_column],title=title) #creating a stacked bar chart with normal plotly function but with 2 comapring values and the desired parameters
    return fig #returing the figure to diplay it


#creating a function for comparing 2 lines in a plot
def double_line(df,x_column,y1_column,y2_column,title): #creating a function with the desired parameters
    fig=px.line(df,x=x_column,y=[y1_column,y2_column],title=title) #creating a double line chart with normal plotly function but with 2 comapring values and the desired parameters
    return fig #returing the figure to diplay it


#creating a function for candlestick chart
def candlestick_chart(df,x_column,open,high,low,close):  #creating a function with the desired parameters
    fig = go.Figure(data=[go.Candlestick(x=df[x_column],  #craeting a candlestick chart with graph_object and the desired parameters
                    open=df[open],
                    high=df[high],
                    low=df[low],
                    close=df[close])])
    return fig #returing the figure to diplay it



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
        plt.figure(figsize=(8, 6))  #adjusting the size of the figure
        fig=sns.countplot(x='Sentiment', data=df) #craeting a count plot
        plt.title('Sentiment Analysis') #titl for the chart
        plt.xlabel('Sentiment')  #labels for x axis
        plt.ylabel('Count') #labels for y axis
        plt.xticks(rotation=45)
        for p in fig.patches:
            fig.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom')  #rotating the labels to better fit and visibility
        return plt




#displaying the column names and their datatypes for user understanding
if df is not None:
    st.sidebar.write(df.dtypes) #displaying the column names and their datatypes to give users and overview of the colums and the datatypes of each column and how to use it

    st.sidebar.write("---------------------------")

    #dividing the page into 4 columns for charts
    with st.container(): #This function creates a container that helps organize content within a Streamlit app. It's useful for grouping related elements together.
        col1, col2 = st.columns(2) #This function splits the container into two columns. The 2 parameter specifies the number of columns to create. In this case, it divides the container into two equal-width columns.

        with col1: #using first column
            chart_type1=st.sidebar.selectbox("Select 1st Chart Type",['Barplot 1','Countplot 1','ScatterPlot 1','LinePlot 1','Histogram 1','BubblePlot 1','Boxplot 1','Voilin Chart 1','Gantt Chart 1','Wordcloud 1','3D Lineplot 1','3D Scatterplot 1','Distplot 1','JointPlot 1','KDE Plot 1','Reg Plot 1','Heatmap 1','Stacked Plot 1','Double Lineplot 1','Candlestick 1','Sentiment Analysis 1'])
            #giving user an option to sleect a chart from the number of charts in sidebar
            if chart_type1=='Barplot 1': #if user selects this chart from the dropdown above
                x_col= st.sidebar.selectbox("Select X Column", df.columns) #select the column for x axis in sidebar
                y_col= st.sidebar.selectbox("Select Y Column", df.columns) #select the column for y axis in sidebar
                x_Axis=st.sidebar.text_input("Enter title for x axis") #enter the title for x axis in sidebar
                y_Axis=st.sidebar.text_input("Enter title for y axis") #enter the title for y axis in sidebar
                plot_title = st.sidebar.text_input("Enter Plot Title", "Bar Plot")  #enter the title for the plot. default is bar plot in sidebar
                plot = create_barplot(df, x_col, y_col,x_Axis,y_Axis, plot_title) #craeting the barplot by calling the above created function for thr bar plot
                st.plotly_chart(plot) #return the chart to display it in the main frame

            elif chart_type1=='Countplot 1':
                x_col= st.sidebar.selectbox("Select X Column", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis")
                y_Axis=st.sidebar.text_input("Enter title for y axis","count")
                plot_title = st.sidebar.text_input("Enter Plot Title", "CountPlot Plot")  
                plot = create_snsplot(df, x_col,x_Axis,y_Axis, plot_title)                        
                st.pyplot(plot)

            elif chart_type1=='ScatterPlot 1':
                x_col= st.sidebar.selectbox("Select X Column", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis")
                y_Axis=st.sidebar.text_input("Enter title for y axis")
                plot_title = st.sidebar.text_input("Enter Plot Title", "Scatter Plot")  
                plot = create_scatterplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot)

            elif chart_type1=='LinePlot 1':
                x_col= st.sidebar.selectbox("Select X Column", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis")
                y_Axis=st.sidebar.text_input("Enter title for y axis")
                plot_title = st.sidebar.text_input("Enter Plot Title", "Line Plot")  
                plot = create_lineplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot) 

            elif chart_type1=='Histogram 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis")
                plot_title=st.sidebar.text_input("Enter title for the plot","Histogram")
                plot=create_histogram(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type1=='BubblePlot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                size_1=st.sidebar.selectbox("Select column for Size",df.columns)
                col=st.sidebar.selectbox("Select column for color",df.columns)
                x_Axis=st.sidebartext_input("Enter title for x axis")
                y_Axis=st.sidebar.text_input("Enter title for Y axis")
                plot_title=st.sidebar.text_input("Enter title for the plot","Bubble Plot")
                plot=create_bubbleplot(df,x_col,y_col,size_1,col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)


            elif chart_type1=='Boxplot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis")
                plot_title=st.sidebar.text_input("Enter title for the plot","Boxplot")
                plot=box_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)


            elif chart_type1=='Voilin Chart 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis")
                plot_title=st.sidebar.text_input("Enter title for the plot","Voilin Plot")
                plot=create_violin_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type1=='Gantt Chart 1':
                x_start_column=st.sidebar.selectbox("Select X Start Column",df.columns)
                x_end_column=st.sidebar.selectbox("Select X End Column",df.columns)
                y_column_name=st.sidebar.selectbox("Select Y Column",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis")
                plot_title=st.sidebar.text_input("Enter title for the plot","Gantt Plot")
                plot=gantt_chart(df,x_start_column,x_end_column,y_column_name,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)
    
            elif chart_type1=='Wordcloud 1':
                review_col=st.sidebar.selectbox("Select the review column",df.columns)
                color=st.sidebar.selectbox("Select the background color",['black','white'])
                title=st.sidebar.text_input("Select the title for wordcloud")
                plot=create_wordcloud(df,review_col,color,title)
                st.pyplot(plot)

            elif chart_type1=='3D Lineplot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","3D Line Plot")
                plot=lineplot_3d(df,x_col,y_col,z_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type1=='3D Scatterplot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column",df.columns)
                col=st.sidebar.selectbox("Select Column for color",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","3D Scatter Plot")
                plot=scatterplot_3d(df,x_col,y_col,z_col,col,plot_title)
                st.plotly_chart(plot)

            elif chart_type1=='Distplot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","DistPlot")
                plot=dist_plot(df,x_col,plot_title)
                st.pyplot(plot)

            elif chart_type1=='JointPlot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","Joint Plot")
                plot=joint_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot)

            elif chart_type1=='KDE Plot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","KDE Plot")
                plot=kde_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type1=='Reg Plot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","Regression Plot")
                plot=reg_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type1=='Heatmap 1':
                plot=create_heatmap(df)
                st.pyplot(plot) 

            elif chart_type1=='Stacked Plot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","Stacked Plot")
                plot=stacked_bar(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type1=='Double Lineplot 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot","Line Plot")
                plot=double_line(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type1=='Candlestick 1':
                x_col=st.sidebar.selectbox("Select X Column",df.columns)
                open_col=st.sidebar.selectbox("Select open prices Column",df.columns)
                high_col=st.sidebar.selectbox("Select high prices Column",df.columns)
                low_col=st.sidebar.selectbox("Select low prices Column",df.columns)
                close_col=st.sidebar.selectbox("Select close prices Column",df.columns)
                plot=candlestick_chart(df,x_col,open_col,high_col,low_col,close_col)
                st.plotly_chart(plot)

            elif chart_type1=='Sentiment Analysis 1':
                x_col=st.selectbox("Select the review column",df.columns)
                dataframe,plot=sentiment_analysis(df,x_col)
                st.pyplot(plot)
        

        st.sidebar.write("----------------------------------------------")
        st.sidebar.write("----------------------------------------------")

        with col2:
            chart_type2=st.sidebar.selectbox("Select 2nd Chart Type",['Barplot 2','Countplot 2','ScatterPlot 2','LinePlot 2','Histogram 2','BubblePlot 2','Boxplot 2','Voilin Chart 2','Gantt Chart 2','Wordcloud 2','3D Lineplot 2','3D Scatterplot 2','Distplot 2','JointPlot 2','KDE Plot 2','Reg Plot 2','Heatmap 2','Stacked Plot 2','Double Lineplot 2','Candlestick 2','Sentiment Analysis 2','None 2'])
            if chart_type2=='Barplot 2':
                x_col= st.sidebar.selectbox("Select X Column for 2nd chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 2nd chart", df.columns)
                x_Axis=st.sidebar.text_input("Enter title for x axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 2nd chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 2nd chart", "Bar Plot")  
                plot = create_barplot(df, x_col, y_col,x_Axis,y_Axis, plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='Countplot 2':
                x_col= st.sidebar.selectbox("Select X Column for 2nd chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 2nd chart","count")
                plot_title = st.sidebar.text_input("Enter Plot Title for 2nd chart", "CountPlot Plot")  
                plot = create_snsplot(df, x_col,x_Axis,y_Axis, plot_title)                        
                st.pyplot(plot)

            elif chart_type2=='ScatterPlot 2':
                x_col= st.sidebar.selectbox("Select X Column for 2nd chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 2nd chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 2nd chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 2nd chart", "Scatter Plot")  
                plot = create_scatterplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot)

            elif chart_type2=='LinePlot 2':
                x_col= st.sidebar.selectbox("Select X Column for 2nd chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 2nd chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 2nd chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 2nd chart", "Line Plot")  
                plot = create_lineplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot) 

            elif chart_type2=='Histogram 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 2nd chart")
                plot_title=st.sidebar.text_input("Enter title  for 2nd chart","Histogram")
                plot=create_histogram(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='BubblePlot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                size_1=st.sidebar.selectbox("Select column for Size for 2nd chart",df.columns)
                col=st.sidebar.selectbox("Select column for color for 2nd chart",df.columns)
                x_Axis=st.sidebartext_input("Enter title for x axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y axis for 2nd chart")
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Bubble Plot")
                plot=create_bubbleplot(df,x_col,y_col,size_1,col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)


            elif chart_type2=='Boxplot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 2nd chart")
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Boxplot")
                plot=box_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)


            elif chart_type2=='Voilin Chart 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 2nd chart")
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Voilin Plot")
                plot=create_violin_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='Gantt Chart 2':
                x_start_column=st.sidebar.selectbox("Select X Start Column for 2nd chart",df.columns)
                x_end_column=st.sidebar.selectbox("Select X End Column for 2nd chart",df.columns)
                y_column_name=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 2nd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 2nd chart")
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Gantt Plot")
                plot=gantt_chart(df,x_start_column,x_end_column,y_column_name,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='Wordcloud 2':
                review_col=st.sidebar.selectbox("Select the review column for 2nd chart",df.columns)
                color=st.sidebar.selectbox("Select the background color for 2nd chart",['black','white'])
                title=st.sidebar.text_input("Select the title for 2nd chart")
                plot=create_wordcloud(df,review_col,color,title)
                st.pyplot(plot)

            elif chart_type2=='3D Lineplot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","3D Line Plot")
                plot=lineplot_3d(df,x_col,y_col,z_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='3D Scatterplot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column for 2nd chart",df.columns)
                col=st.sidebar.selectbox("Select Column for color for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","3D Scatter Plot")
                plot=scatterplot_3d(df,x_col,y_col,z_col,col,plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='Distplot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","DistPlot")
                plot=dist_plot(df,x_col,plot_title)
                st.pyplot(plot)

            elif chart_type2=='JointPlot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Joint Plot")
                plot=joint_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot)

            elif chart_type2=='KDE Plot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot for 2nd chart","KDE Plot")
                plot=kde_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type2=='Reg Plot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Regression Plot")
                plot=reg_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type2=='Heatmap 2':
                plot=create_heatmap(df)
                st.pyplot(plot) 

            elif chart_type2=='Stacked Plot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column for 2nd chart",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Stacked Plot")
                plot=stacked_bar(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='Double Lineplot 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column for 2nd chart",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column for 2nd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 2nd chart","Line Plot")
                plot=double_line(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type2=='Candlestick 2':
                x_col=st.sidebar.selectbox("Select X Column for 2nd chart",df.columns)
                open_col=st.sidebar.selectbox("Select open prices Column for 2nd chart",df.columns)
                high_col=st.sidebar.selectbox("Select high prices Column for 2nd chart",df.columns)
                low_col=st.sidebar.selectbox("Select low prices Column for 2nd chart",df.columns)
                close_col=st.sidebar.selectbox("Select close prices Column for 2nd chart",df.columns)
                plot=candlestick_chart(df,x_col,open_col,high_col,low_col,close_col)
                st.plotly_chart(plot)

            elif chart_type2=='Sentiment Analysis 2':
                x_col=st.selectbox("Select the review column",df.columns)
                dataframe,plot=sentiment_analysis(df,x_col)
                st.pyplot(plot)

            elif chart_type2=='None 2':
                pass

        st.sidebar.write("---------------------------------------------------")
        st.sidebar.write("---------------------------------------------------")

    with st.container():
        col3, col4 = st.columns(2)
        with col3:
            chart_type3=st.sidebar.selectbox("Select 3rd Chart Type",['Barplot 3','Countplot 3','ScatterPlot 3','LinePlot 3','Histogram 3','BubblePlot 3','Boxplot 3','Voilin Chart 3','Gantt Chart 3','Wordcloud 3','3D Lineplot 3','3D Scatterplot 3','Distplot 3','JointPlot 3','KDE Plot 3','Reg Plot 3','Heatmap 3','Stacked Plot 3','Double Lineplot 3','Candlestick 3','Sentiment Analysis 3','None 3'])
            if chart_type3=='Barplot 3':
                x_col= st.sidebar.selectbox("Select X Column for 3rd chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 3rd chart", df.columns)
                x_Axis=st.sidebar.text_input("Enter title for x axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 3rd chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 3rd chart", "Bar Plot")  
                plot = create_barplot(df, x_col, y_col,x_Axis,y_Axis, plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='Countplot 3':
                x_col= st.sidebar.selectbox("Select X Column for 3rd chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 3rd chart","count")
                plot_title = st.sidebar.text_input("Enter Plot Title for 3rd chart", "CountPlot Plot")  
                plot = create_snsplot(df, x_col,x_Axis,y_Axis, plot_title)                        
                st.pyplot(plot)

            elif chart_type3=='ScatterPlot 3':
                x_col= st.sidebar.selectbox("Select X Column for 3rd chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 3rd chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 3rd chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 3rd chart", "Scatter Plot")  
                plot = create_scatterplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot)

            elif chart_type3=='LinePlot 3':
                x_col= st.sidebar.selectbox("Select X Column for 3rd chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 3rd chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 3rd chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 3rd chart", "Line Plot")  
                plot = create_lineplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot) 

            elif chart_type3=='Histogram 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 3rd chart")
                plot_title=st.sidebar.text_input("Enter title  for 3rd chart","Histogram")
                plot=create_histogram(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='BubblePlot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                size_1=st.sidebar.selectbox("Select column for Size for 3rd chart",df.columns)
                col=st.sidebar.selectbox("Select column for color for 3rd chart",df.columns)
                x_Axis=st.sidebartext_input("Enter title for x axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y axis for 3rd chart")
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Bubble Plot")
                plot=create_bubbleplot(df,x_col,y_col,size_1,col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)


            elif chart_type3=='Boxplot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 3rd chart")
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Boxplot")
                plot=box_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)



            elif chart_type3=='Voilin Chart 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 3rd chart")
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Voilin Plot")
                plot=create_violin_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='Gantt Chart 3':
                x_start_column=st.sidebar.selectbox("Select X Start Column for 3rd chart",df.columns)
                x_end_column=st.sidebar.selectbox("Select X End Column for 3rd chart",df.columns)
                y_column_name=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 3rd chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 3rd chart")
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Gantt Plot")
                plot=gantt_chart(df,x_start_column,x_end_column,y_column_name,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='Wordcloud 3':
                review_col=st.sidebar.selectbox("Select the review column for 3rd chart",df.columns)
                color=st.sidebar.selectbox("Select the background color for 3rd chart",['black','white'])
                title=st.sidebar.text_input("Select the title for 3rd chart")
                plot=create_wordcloud(df,review_col,color,title)
                st.pyplot(plot)

            elif chart_type3=='3D Lineplot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","3D Line Plot")
                plot=lineplot_3d(df,x_col,y_col,z_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='3D Scatterplot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column for 3rd chart",df.columns)
                col=st.sidebar.selectbox("Select Column for color for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","3D Scatter Plot")
                plot=scatterplot_3d(df,x_col,y_col,z_col,col,plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='Distplot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","DistPlot")
                plot=dist_plot(df,x_col,plot_title)
                st.pyplot(plot)

            elif chart_type3=='JointPlot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Joint Plot")
                plot=joint_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot)

            elif chart_type3=='KDE Plot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot for 3rd chart","KDE Plot")
                plot=kde_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type3=='Reg Plot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Regression Plot")
                plot=reg_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type3=='Heatmap 3':
                plot=create_heatmap(df)
                st.pyplot(plot) 

            elif chart_type3=='Stacked Plot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column for 3rd chart",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Stacked Plot")
                plot=stacked_bar(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='Double Lineplot 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column for 3rd chart",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column for 3rd chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 3rd chart","Line Plot")
                plot=double_line(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type3=='Candlestick 3':
                x_col=st.sidebar.selectbox("Select X Column for 3rd chart",df.columns)
                open_col=st.sidebar.selectbox("Select open prices Column for 3rd chart",df.columns)
                high_col=st.sidebar.selectbox("Select high prices Column for 3rd chart",df.columns)
                low_col=st.sidebar.selectbox("Select low prices Column for 3rd chart",df.columns)
                close_col=st.sidebar.selectbox("Select close prices Column for 3rd chart",df.columns)
                plot=candlestick_chart(df,x_col,open_col,high_col,low_col,close_col)
                st.plotly_chart(plot)

            elif chart_type3=='Sentiment Analysis 3':
                x_col=st.selectbox("Select the review column",df.columns)
                dataframe,plot=sentiment_analysis(df,x_col)
                st.pyplot(plot)

            elif chart_type3=='None 3':
                pass

        st.sidebar.write("-------------------------------------------------------------------")
        st.sidebar.write("--------------------------------------------------------------------")
    
        with col4:
            chart_type4=st.sidebar.selectbox("Select 4th Chart Type",['Barplot 4','Countplot 4','ScatterPlot 4','LinePlot 4','Histogram 4','BubblePlot 4','Boxplot 4','Voilin Chart 4','Gantt Chart 4','Wordcloud 4','3D Lineplot 4','3D Scatterplot 4','Distplot 4','JointPlot 4','KDE Plot 4','Reg Plot 4','Heatmap 4','Stacked Plot 4','Double Lineplot 4','Candlestick 4','Sentiment Analysis 4','None 4'])
            if chart_type4=='Barplot 4':
                x_col= st.sidebar.selectbox("Select X Column for 4th chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 4th chart", df.columns)
                x_Axis=st.sidebar.text_input("Enter title for x axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 4th chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 4th chart", "Bar Plot")  
                plot = create_barplot(df, x_col, y_col,x_Axis,y_Axis, plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='Countplot 4':
                x_col= st.sidebar.selectbox("Select X Column for 4th chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 4th chart","count")
                plot_title = st.sidebar.text_input("Enter Plot Title for 4th chart", "CountPlot Plot")  
                plot = create_snsplot(df, x_col,x_Axis,y_Axis, plot_title)                        
                st.pyplot(plot)

            elif chart_type4=='ScatterPlot 4':
                x_col= st.sidebar.selectbox("Select X Column for 4th chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 4th chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 4th chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 4th chart", "Scatter Plot")  
                plot = create_scatterplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot)

            elif chart_type4=='LinePlot 4':
                x_col= st.sidebar.selectbox("Select X Column for 4th chart", df.columns)
                y_col= st.sidebar.selectbox("Select Y Column for 4th chart", df.columns)              
                x_Axis=st.sidebar.text_input("Enter title for x axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for y axis for 4th chart")
                plot_title = st.sidebar.text_input("Enter Plot Title for 4th chart", "Line Plot")  
                plot = create_lineplot(df, x_col,y_col,x_Axis,y_Axis, plot_title)                        
                st.plotly_chart(plot) 

            elif chart_type4=='Histogram 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 4th chart")
                plot_title=st.sidebar.text_input("Enter title  for 4th chart","Histogram")
                plot=create_histogram(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='BubblePlot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                size_1=st.sidebar.selectbox("Select column for Size for 4th chart",df.columns)
                col=st.sidebar.selectbox("Select column for color for 4th chart",df.columns)
                x_Axis=st.sidebartext_input("Enter title for x axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for Y axis for 4th chart")
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Bubble Plot")
                plot=create_bubbleplot(df,x_col,y_col,size_1,col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)


            elif chart_type4=='Boxplot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 4th chart")
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Boxplot")
                plot=box_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)



            elif chart_type4=='Voilin Chart 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 4th chart")
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Voilin Plot")
                plot=create_violin_plot(df,x_col,y_col,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='Gantt Chart 4':
                x_start_column=st.sidebar.selectbox("Select X Start Column for 4th chart",df.columns)
                x_end_column=st.sidebar.selectbox("Select X End Column for 4th chart",df.columns)
                y_column_name=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                x_Axis=st.sidebar.text_input("Enter title for X axis for 4th chart")
                y_Axis=st.sidebar.text_input("Enter title for Y Axis for 4th chart")
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Gantt Plot")
                plot=gantt_chart(df,x_start_column,x_end_column,y_column_name,x_Axis,y_Axis,plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='Wordcloud 4':
                review_col=st.sidebar.selectbox("Select the review column for 4th chart",df.columns)
                color=st.sidebar.selectbox("Select the background color for 4th chart",['black','white'])
                title=st.sidebar.text_input("Select the title for 4th chart")
                plot=create_wordcloud(df,review_col,color,title)
                st.pyplot(plot)

            elif chart_type4=='3D Lineplot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 4th chart","3D Line Plot")
                plot=lineplot_3d(df,x_col,y_col,z_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='3D Scatterplot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                z_col=st.sidebar.selectbox("Select Z Column for 4th chart",df.columns)
                col=st.sidebar.selectbox("Select Column for color for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 4th chart","3D Scatter Plot")
                plot=scatterplot_3d(df,x_col,y_col,z_col,col,plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='Distplot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 4th chart","DistPlot")
                plot=dist_plot(df,x_col,plot_title)
                st.pyplot(plot)

            elif chart_type4=='JointPlot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Joint Plot")
                plot=joint_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot)

            elif chart_type4=='KDE Plot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for the plot for 4th chart","KDE Plot")
                plot=kde_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type4=='Reg Plot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y_col=st.sidebar.selectbox("Select Y Column for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Regression Plot")
                plot=reg_plot(df,x_col,y_col,plot_title)
                st.pyplot(plot) 

            elif chart_type4=='Heatmap 4':
                plot=create_heatmap(df)
                st.pyplot(plot) 

            elif chart_type4=='Stacked Plot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column for 4th chart",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Stacked Plot")
                plot=stacked_bar(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='Double Lineplot 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                y1_col=st.sidebar.selectbox("Select first comparison Column for 4th chart",df.columns)
                y2_col=st.sidebar.selectbox("Select second comparison Column for 4th chart",df.columns)
                plot_title=st.sidebar.text_input("Enter title for 4th chart","Line Plot")
                plot=double_line(df,x_col,y1_col,y2_col,plot_title)
                st.plotly_chart(plot)

            elif chart_type4=='Candlestick 4':
                x_col=st.sidebar.selectbox("Select X Column for 4th chart",df.columns)
                open_col=st.sidebar.selectbox("Select open prices Column for 4th chart",df.columns)
                high_col=st.sidebar.selectbox("Select high prices Column for 4th chart",df.columns)
                low_col=st.sidebar.selectbox("Select low prices Column for 4th chart",df.columns)
                close_col=st.sidebar.selectbox("Select close prices Column for 4th chart",df.columns)
                plot=candlestick_chart(df,x_col,open_col,high_col,low_col,close_col)
                st.plotly_chart(plot)

            elif chart_type4=='Sentiment Analysis 4':
                x_col=st.selectbox("Select the review column",df.columns)
                dataframe,plot=sentiment_analysis(df,x_col)
                st.pyplot(plot)

            elif chart_type4=='None 4':
                pass
