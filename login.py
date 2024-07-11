import streamlit as st  #for web application using python
import sqlite3    #working with SQLite database
import hashlib    #password hashing in database for security and protection of data
import subprocess    #To connect to other python page by redirecting it
from PIL import Image #python image libarry to load iamges
import base64 # encoding binary data into ASCII characters


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
    st.markdown(page_bg_img, unsafe_allow_html=True)  #second parameter allows to ender html content safely

# Use this function to set the background image of your app
set_png_as_page_bg('bg2.png')



st.title(':white[CipherQuest]')
# This name means embark on the quest to decode the hidden insight in data


img=Image.open("logo_app.png")
wid=630
heigh=200
img_resize=img.resize((wid,heigh))
st.image(img_resize)




#hashing is a process of taking input as a string mostly passwords and converting it to a fixed sized string for security purpose
#.sha256-cryptographic hash function that converts the password into a long string
#encode method converts a string into bytes that is numeric representation
#hexadigest is a numeral system that uses 0-9 and A-F to represent value. It contains 16 digits. heer it is used to convert binary data that is encoded to hexadigest
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

#creating a function to create a table in database
def create_table():
    with sqlite3.connect('app.db') as conn: #establishing a connection
        cursor = conn.cursor() #creating cursor object that will execute the query
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS owner_database (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                First_Name TEXT,
                Last_Name TEXT,
                Email_ID TEXT,
                Phone_no TEXT,
                Password TEXT
            )
        ''')  #creating a table named application_record_db and adding the attributes in it


#creating a login function
def login():
    st.title("Already have an account? Login")
# adding a title
    username = st.text_input("Enter Email ID")
    password = st.text_input("Enter password", type="password")
#taking password and username that is the email id from the user
    if st.button("Login"):
        with sqlite3.connect('app.db') as conn:  #again establishing a connection
            cursor = conn.cursor()  #creating a cursor to execute a query
            query = "SELECT * FROM owner_database WHERE Email_ID=? AND Password=?"
            #this is the query taht we will execute. and this will check for email id and password
            hashed_password = hash_password(password)  #the password which the user has entered will be hashed here
            cursor.execute(query, (username, hashed_password))  #execute the above query and add teh parametrs as username and password to this query which are defined by the user above
            user = cursor.fetchone() # fetch the above record if teh entered details amtch the already existing details in the database and store it in a new variable anmed user

        if user:  #if the variable user craeted above is not null which means the values amtch with the alraedy existing record
            st.success("Logged in Successfully!") #this message will be displayed
            st.balloons() #a streamlit function that will float balloons
            #this entire function will be runed on a click of the login button , so on clicking the button this function will run
            subprocess.run(["streamlit", "run", "final_change.py"],shell=True) 
            #after clicking using the subprocess function we will run the first_app.py page which contains our main app. so this page will be directed to that main app page
        else:
            st.error("Invalid Credentials! Try Again!") #if the details entered by user dont match the already exisiting records then this message will be displayed

#creating a new function to create a new account if the suer account dont exist before
def create_account():
    st.title("Create Account") #display this text

    first = st.text_input("Enter First Name", key="first_name_input")  #take first name as input from user
    last = st.text_input("Enter Last Name", key="last_name_input")    #take last name as input from user
    email_id = st.text_input("Enter Email ID", key="email_input")   #take email as input from user
    phone_no = st.text_input("Enter phone number",key="number_input")
    password = st.text_input("Enter Password", type='password', key="password_input")  #take password as input from user
    con_pass = st.text_input("Confirm Password", type='password', key="con_password_input")   #confirm the password , usaaer will renter

    if password == con_pass and st.button("Create"): #if the 2 entered passwords match on hitting the create button users acc gets created and the details are stored in the database
        st.success("Passwords Matched!")
        hashed_password = hash_password(password)#convert password to hashed password for security
        with sqlite3.connect('app.db') as conn: #establish a connection
            cursor = conn.cursor()  #create a cursor object
            query = "INSERT INTO owner_database (First_Name, Last_Name, Email_ID, Phone_no, Password) VALUES (?, ?, ?, ?, ?)"   #this is the query for inserting the data
            cursor.execute(query, (first, last, email_id, phone_no, hashed_password)) #insert the values in the datatabse that are entered by the user by query execution
            conn.commit()  #commit the chnages
        st.success("Account Created Successfully!!! Login with your credentials")   #after entering details in database this gets displayed
        st.balloons()  #balloons are floated 
    elif password != con_pass:  #if both passwors do not match, an error message is generated
        st.error("Passwords do not match! Try Again!")

def main():
    #craeting a main function 

    choice = st.radio("Select an option:", ("Login", "Create Account"))

#create a radio button that provides 2 option one is to craete an account for new user and second is to login for existing user
    if choice == "Login":  #if the radio option selected by the user is login
        login()  #run the login function created above
    elif choice == "Create Account": #if the radio option selected by user is craete account
        create_account()  #run the craete acc function craeted above

if __name__ == "__main__": #checks if it is run directly (the python script) and not imported as a module
    create_table()  #craeting the database table
    main()  #running the main function
