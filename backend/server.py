# Import necessary libraries
import flask
from flask import jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from ml_models.chabot_response import get_model_response
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = flask.Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing) for the app
CORS(app)

# Function to establish a connection to the PostgreSQL database
def get_database():
    db_url = os.getenv("db_url")  # Get database URL from environment variables
    return psycopg2.connect(db_url)  # Return the database connection

# Route to handle user registration data via POST request
@app.route("/", methods=["POST"])
def post_register_data():
    if request.is_json:  # Check if the request contains JSON data
        data = request.get_json()  # Parse JSON data from the request
        print(data)  # Print the received data for debugging

        # Extract user details from the JSON data
        firstname = data.get("firstname")
        lastname = data.get("lastname")
        email = data.get("email")
        password = data.get("password")
        confirmPassword = data.get("confirmPassword")

        # Connect to the database
        connection = get_database()
        print("connection established successfully!!")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # Check if the 'UserAccount' table exists in the database
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = 'useraccount'
            )
            """
        )
        table_exists = cursor.fetchone()[0]  # Fetch the result of the query

        # If the table does not exist, create it
        if not table_exists:
            cursor.execute(
                """
                CREATE TABLE UserAccount(
                   user_id SERIAL PRIMARY KEY,
                    firstname varchar(50) NOT NULL,
                    lastname varchar(50) NOT NULL,
                    email varchar(100) NOT NULL UNIQUE,
                    password varchar(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """
            )
            print("The table is created successfully!! ")

        # Insert the user's registration data into the 'UserAccount' table
        cursor.execute(
            """ 
            INSERT INTO UserAccount(firstname, lastname, email, password)
                       VALUES(%s,%s,%s,%s)
        """,
            (firstname, lastname, email, password),
        )

        # Commit the transaction to save changes to the database
        connection.commit()
        # Close the cursor and database connection
        cursor.close()
        connection.close()
        print("Value is inserted")

    # Return the received data as a JSON response
    return jsonify(data)

# Route to handle user login data via POST or GET request
@app.route("/post_login_data", methods=["POST", "GET"])
def post_login_data():
    if request.is_json and request.method == "POST":  # Check if the request is JSON and a POST request
        data = request.get_json()  # Parse JSON data from the request
        email = data.get("email")
        password = data.get("password")
        
        # Connect to the database
        connection = get_database()
        cursor = connection.cursor()

        # Query the database to check if the email exists
        cursor.execute(
            """ 
            SELECT * FROM UserAccount WHERE email = %s
        """,
            (email,),
        )
        user = cursor.fetchone()  # Fetch the user record
        print(user)

        # If the user exists, check if the password matches
        if user:
            stored_password = user[4]  # Assuming password is at index 4
            if password == stored_password:
                # Return a success message and user details if login is successful
                return jsonify({
                    "message": "Login successful", 
                    "user": { 
                        "id": user[0],  # user_id
                        "name": f"{user[1]} {user[2]}"  # firstname and lastname combined
                    }
                })
            else:
                # Return an error message if credentials are invalid
                return jsonify({"message": "Invalid credentials"}), 401
        else:
            # Return an error message if the user is not found
            return jsonify({"message": "User not found"}), 404

    # Return an error message for invalid requests
    return jsonify({"message": "Invalid request"}), 400

# Route to handle user input and chatbot interaction via POST request
@app.route("/post_userinput", methods=["POST"])
def post_userinput():
    if request.is_json:  # Check if the request contains JSON data
        data = request.get_json()  # Parse JSON data from the request
        user_input = data.get("input")
        user_id = data.get("user_id")  # Get the user_id

        print(f"The user input is: {user_input}")

        # Get the chatbot's response using the ML model
        chabot_response = get_model_response(user_input)

        # Connect to the database
        connection = get_database()
        print("connection established successfully!!")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # Check if the 'interactionlog' table exists in the database
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = 'interactionlog'
            )
        """
        )
        table_exists = cursor.fetchone()[0]  # Fetch the result of the query

        # If the table does not exist, create it
        if not table_exists:
            cursor.execute(
                """
                CREATE TABLE interactionlog(
                   interaction_id SERIAL PRIMARY KEY,
                    user_input varchar(1000),
                    llm_response varchar(1000),
                    user_id INTEGER REFERENCES UserAccount(user_id) -- Reference to user_id
                )
            """
            )
            print("The table is created successfully!! ")

        # Insert the user input and chatbot response into the 'interactionlog' table
        cursor.execute(
            """ 
            INSERT INTO interactionlog(user_input, llm_response, user_id)
                       VALUES(%s,%s,%s)
        """,
            (user_input, chabot_response, user_id),
        )

        # Commit the transaction to save changes to the database
        connection.commit()
        # Close the cursor and database connection
        cursor.close()
        connection.close()
        print("Value is inserted")

        # Prepare the response to be returned
        response = {
            "user_input": f"{user_input}",
            "chatbot_response": f"{chabot_response}",
        }

        # Return the response as JSON
        return jsonify(response)

# Route to handle user rating submission via POST request
@app.route("/submit_rating", methods=["POST"])
def submit_rating():
    if request.is_json:  # Check if the request contains JSON data
        data = request.get_json()  # Parse JSON data from the request
        rating = data.get("rating")
        user_id = data.get("user_id")  # Get the user_id

        # Connect to the database
        connection = get_database()
        cursor = connection.cursor()

        # Create the 'Ratings' table if it does not exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Ratings (
                rating_id SERIAL PRIMARY KEY,
                rating INTEGER NOT NULL,
                user_id INTEGER REFERENCES UserAccount(user_id), -- Reference to user_id
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Insert the user's rating into the 'Ratings' table
        cursor.execute(
            "INSERT INTO Ratings (rating, user_id) VALUES (%s, %s)",
            (rating, user_id),
        )
        # Commit the transaction to save changes to the database
        connection.commit()

        # Close the cursor and database connection
        cursor.close()
        connection.close()

        # Return a success message
        return jsonify({"message": "Rating submitted successfully"}), 200

    # Return an error message for invalid requests
    return jsonify({"error": "Invalid request"}), 400

# Route to handle user feedback submission via POST request
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    if request.is_json:  # Check if the request contains JSON data
        data = request.get_json()  # Parse JSON data from the request
        feedback_message = data.get("feedback")
        user_id = data.get("user_id")  # Get the user_id

        # Connect to the database
        connection = get_database()
        cursor = connection.cursor()

        # Create the 'Feedback' table if it does not exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Feedback (
                feedback_id SERIAL PRIMARY KEY,
                feedback_message TEXT NOT NULL,
                user_id INTEGER REFERENCES UserAccount(user_id), -- Reference to user_id
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Check if feedback already exists for the user
        cursor.execute(
            """
            SELECT feedback_id FROM Feedback WHERE user_id = %s
            """,
            (user_id,),
        )
        existing_feedback = cursor.fetchone()  # Fetch the result of the query

        if existing_feedback:
            # If feedback exists, update the feedback message
            cursor.execute(
                """
                UPDATE Feedback
                SET feedback_message = %s, created_at = CURRENT_TIMESTAMP
                WHERE feedback_id = %s
                """,
                (feedback_message, existing_feedback[0]),
            )
            print("Feedback updated successfully!")
        else:
            # If no feedback exists, insert new feedback
            cursor.execute(
                """
                INSERT INTO Feedback (feedback_message, user_id)
                VALUES (%s, %s)
                """,
                (feedback_message, user_id),
            )
            print("Feedback inserted successfully!")

        # Commit the transaction to save changes to the database
        connection.commit()
        # Close the cursor and database connection
        cursor.close()
        connection.close()

        # Return a success message
        return jsonify({"message": "Feedback submitted successfully"}), 200

    # Return an error message for invalid requests
    return jsonify({"error": "Invalid request"}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Start the Flask app in debug mode