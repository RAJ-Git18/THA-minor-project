import flask
from flask import jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from ml_models.chabot_response import get_model_response

import psycopg2

load_dotenv()

app = flask.Flask(__name__)
CORS(app)

# database connection
def get_database():
    db_url = os.getenv("db_url")
    return psycopg2.connect(db_url)

@app.route("/", methods=["POST"])
def post_register_data():
    if request.is_json:
        data = request.get_json()
        print(data)

        firstname = data.get("firstname")
        lastname = data.get("lastname")
        email = data.get("email")
        password = data.get("password")
        confirmPassword = data.get("confirmPassword")

        # connecting to the database
        connection = get_database()
        print("connection established successfully!!")

        # creating a cursor
        cursor = connection.cursor()

        # executing the sql queries
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = 'useraccount'
            )
            """
        )
        table_exists = cursor.fetchone()[0]

        # creating the table if it doesn't exist
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

        # inserting the values
        cursor.execute(
            """ 
            INSERT INTO UserAccount(firstname, lastname, email, password)
                       VALUES(%s,%s,%s,%s)
        """,
            (firstname, lastname, email, password),
        )

        # commit the changes
        connection.commit()
        cursor.close()
        connection.close()
        print("Value is inserted")

    return jsonify(data)

@app.route("/post_login_data", methods=["POST", "GET"])
def post_login_data():
    if request.is_json and request.method == "POST":
        data = request.get_json()
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
        user = cursor.fetchone()
        print(user)

        # If the user exists
        if user:
            stored_password = user[4]  # Assuming password is at index 4
            if password == stored_password:
                return jsonify({
                    "message": "Login successful", 
                    "user": { 
                        "id": user[0],  # user_id
                        "name": f"{user[1]} {user[2]}"  # firstname and lastname combined
                    }
                })
            else:
                return jsonify({"message": "Invalid credentials"}), 401
        else:
            return jsonify({"message": "User not found"}), 404

    return jsonify({"message": "Invalid request"}), 400


@app.route("/post_userinput", methods=["POST"])
def post_userinput():
    if request.is_json:
        data = request.get_json()
        user_input = data.get("input")
        user_id = data.get("user_id")  # Get the user_id

        print(f"The user input is: {user_input}")

        chabot_response = get_model_response(user_input)

        connection = get_database()
        print("connection established successfully!!")

        # creating a cursor
        cursor = connection.cursor()

        # executing the sql queries
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = 'interactionlog'
            )
        """
        )
        table_exists = cursor.fetchone()[0]

        # creating the table if it doesn't exist
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

        # inserting the values
        # cursor.execute(
        #     """ 
        #     INSERT INTO interactionlog(user_input, llm_response, user_id)
        #                VALUES(%s,%s,%s)
        # """,
        #     (user_input, chabot_response, user_id),
        # )

        # commit the changes
        connection.commit()
        cursor.close()
        connection.close()
        print("Value is inserted")

        response = {
            "user_input": f"{user_input}",
            "chatbot_response": f"{chabot_response}",
        }

        return jsonify(response)


@app.route("/submit_rating", methods=["POST"])
def submit_rating():
    if request.is_json:
        data = request.get_json()
        rating = data.get("rating")
        user_id = data.get("user_id")  # Get the user_id

        connection = get_database()
        cursor = connection.cursor()

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

        cursor.execute(
            "INSERT INTO Ratings (rating, user_id) VALUES (%s, %s)",
            (rating, user_id),
        )
        connection.commit()

        cursor.close()
        connection.close()

        return jsonify({"message": "Rating submitted successfully"}), 200

    return jsonify({"error": "Invalid request"}), 400


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    if request.is_json:
        data = request.get_json()
        feedback_message = data.get("feedback")
        user_id = data.get("user_id")  # Get the user_id

        connection = get_database()
        cursor = connection.cursor()

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
        existing_feedback = cursor.fetchone()

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

        connection.commit()
        cursor.close()
        connection.close()

        return jsonify({"message": "Feedback submitted successfully"}), 200

    return jsonify({"error": "Invalid request"}), 400


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
