import sqlite3  #working with SQLite database
import streamlit as st  #python library for creating web application
from io import StringIO #facilitates working with streams of data.It allows to treat string as a file.enabling you to perform file-like operations on string data.

#to make the display on the output wide we mneed to change the configuration of the page
st.set_page_config(layout="wide")

#creating a function
def display_all_data():
    conn = sqlite3.connect('app.db')  #estblishing a connection to the database
    cursor = conn.cursor()   #craeet a cursor object, it acts as a ponter and allows to execute teh query given by us
    
    query = "SELECT * from owner_database"  #the query that is to be executed is stored in a variable
    #this query selects all the record from the app_database table
    cursor.execute(query)  #executing the query using cursor object
    all_data = cursor.fetchall()  #fetching all the data

    # Display the fetched data
    for row in all_data:
        st.write(row)
#if we use print the output is displayed in console and if we use st.write teh output is displayed in app
    conn.close()  #closing connection

#call the function to display the data
display_all_data()
