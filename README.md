# Interactive Chatbot for Mental Wellbeing

**Interactive Chatbot for Mental Wellbeing** is an innovative AI-powered platform designed to provide empathetic, contextually aware mental health support. Built as a single-repository project, it integrates a TypeScript-based frontend and a Flask-powered backend, leveraging GPT-2 fine-tuned on mental health datasets and PostgreSQL for robust database management. This project was developed to explore the intersection of AI and emotional wellbeing, offering users a safe space to express their concerns and receive supportive responses.

![Chatbot User Interface](screenshots/chatbot_response.png)

## Team Members
- **Amrita Shahi** ([@amritashahi](https://github.com/amritashahi))  
- **Pranil Parajuli** ([@praniil](https://github.com/praniil))  
- **Raj Simkhada** ([@RAJ-Git18](https://github.com/RAJ-Git18))  
- **Sugam Dahal** ([@sugam24](https://github.com/sugam24))  
*Department of Electronics and Computer Engineering, Tribhuvan University, Institute of Engineering, Thapathali Campus, Kathmandu*

## Project Motivation
The rapid evolution of artificial intelligence (AI) technologies has opened avenues for addressing critical social challenges, including emotional isolation and loneliness. In a world where people often feel disconnected despite being digitally connected, there is a rising need for innovative solutions that foster emotional support. Many individuals, especially those who live alone or face social challenges, may lack meaningful interactions, resulting in adverse effects on mental health. The motivation for this project arises from the potential of AI to act as a bridge in providing companionship to those in need. Unlike conventional chatbots that deliver transactional responses, a human-companion chatbot focuses on empathy and understanding. This aligns with the increasing demand for AI systems that can analyze emotions and adapt to user sentiments.

Recent advancements in natural language processing (NLP), voice sentiment analysis, and transformer-based models like GPT have made it possible to create more human-like interactions. By leveraging these technologies, the project aims to develop a system capable of understanding not only what users say but also how they feel. This approach aspires to bring a level of emotional intelligence to AI systems that has been largely absent in existing solutions.

The project is also inspired by the societal need for accessible mental health support tools. By acting as a virtual companion, the chatbot can help individuals alleviate feelings of loneliness, provide a listening ear, and stimulate meaningful conversations. This motivation underpins the belief that AI, when designed thoughtfully, can contribute positively to mental and emotional wellbeing.

## Project Overview
The **Interactive Chatbot for Mental Wellbeing** combines a modern frontend built with TypeScript and React, a scalable Flask backend, and a PostgreSQL database to deliver a seamless user experience. The chatbot is fine-tuned on diverse mental health datasets from Hugging Face, enabling it to generate empathetic and relevant responses. Key functionalities include user registration/login, real-time chat, feedback submission, and rating capabilities, all tied together with a robust database architecture.

### Key Features
- **Empathetic Chatbot**: Powered by a fine-tuned GPT-2 model, the chatbot provides human-like, supportive responses to user inputs on topics like anxiety, stress, and loneliness.
- **User Management**: Secure registration, login, and profile tracking with PostgreSQL-backed storage.
- **Real-Time Interaction**: Users can engage in dynamic conversations, with responses logged for analysis.
- **Feedback & Rating System**: Collects user feedback and ratings to improve chatbot performance.
- **Predefined Questions**: Offers a list of frequent mental health queries to guide users.
- **Database-Driven Insights**: Stores interactions, feedback, and ratings for future enhancements.

## Tech Stack
- **Frontend**: TypeScript, React, Axios, Tailwind CSS 
- **Backend**: Flask, Flask-CORS, Python
- **Database**: PostgreSQL
- **Machine Learning**: GPT-2 (fine-tuned), Transformers, PyTorch
- **Other Tools**: python-dotenv, psycopg2-binary

## Database Schema
The database is a cornerstone of this project, ensuring data persistence, user tracking, and interaction logging. Below is the schema derived from the `server.py` file, covering all four tables:

### 1. `UserAccount`
Stores user registration and authentication details.
- **user_id**: `SERIAL PRIMARY KEY` - Unique identifier for each user.
- **firstname**: `VARCHAR(50) NOT NULL` - User’s first name.
- **lastname**: `VARCHAR(50) NOT NULL` - User’s last name.
- **email**: `VARCHAR(100) NOT NULL UNIQUE` - User’s email (unique constraint).
- **password**: `VARCHAR(50) NOT NULL` - User’s password (stored as plain text in this implementation; consider hashing in production).
- **created_at**: `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` - Account creation timestamp.
- **last_login**: `TIMESTAMP` - Last login timestamp (nullable).

**SQL Definition**:
```sql
CREATE TABLE UserAccount (
    user_id SERIAL PRIMARY KEY,
    firstname VARCHAR(50) NOT NULL,
    lastname VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

### 2. `InteractionLog`
Logs all chatbot interactions for analysis and improvement.
- **interaction_id**: `SERIAL PRIMARY KEY` - Unique identifier for each interaction.
- **user_input**: `VARCHAR(500)` - User’s input to the chatbot.
- **llm_response**: `VARCHAR(500)` - Chatbot’s generated response.
- **user_id**: `INTEGER REFERENCES UserAccount(user_id)` - Foreign key linking to the user.

**SQL Definition**:
```sql
CREATE TABLE InteractionLog (
    interaction_id SERIAL PRIMARY KEY,
    user_input VARCHAR(500),
    llm_response VARCHAR(500),
    user_id INTEGER REFERENCES UserAccount(user_id)
);
```

### 3. `Ratings`
Stores user ratings for the chatbot experience.
- **rating_id**: `SERIAL PRIMARY KEY` - Unique identifier for each rating.
- **rating**: `INTEGER NOT NULL` - Rating value (e.g., 1-5).
- **user_id**: `INTEGER REFERENCES UserAccount(user_id)` - Foreign key linking to the user.
- **created_at**: `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` - Rating submission timestamp.

**SQL Definition**:
```sql
CREATE TABLE Ratings (
    rating_id SERIAL PRIMARY KEY,
    rating INTEGER NOT NULL,
    user_id INTEGER REFERENCES UserAccount(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. `Feedback`
Captures detailed user feedback for iterative improvements.
- **feedback_id**: `SERIAL PRIMARY KEY` - Unique identifier for each feedback entry.
- **feedback_message**: `TEXT NOT NULL` - User’s feedback text.
- **user_id**: `INTEGER REFERENCES UserAccount(user_id)` - Foreign key linking to the user.
- **created_at**: `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` - Feedback submission timestamp.

**SQL Definition**:
```sql
CREATE TABLE Feedback (
    feedback_id SERIAL PRIMARY KEY,
    feedback_message TEXT NOT NULL,
    user_id INTEGER REFERENCES UserAccount(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Notes**:
- Foreign keys (`user_id`) ensure referential integrity between tables.
- The schema dynamically creates tables if they don’t exist, as implemented in `server.py`.
- Consider increasing `VARCHAR` limits (e.g., from 500 to 1000) for `InteractionLog` fields if longer inputs/responses are expected.
- Passwords should ideally be hashed (e.g., using `bcrypt`) for security in a production environment.

## Prerequisites
- **Node.js** and **npm** (for frontend)
- **Python 3.8+** (for backend)
- **PostgreSQL** (for database)
- **CUDA-enabled GPU** (optional, for model training)
- A modern web browser

## Installation
Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[YourUsername]/interactive-chatbot-for-mental-wellbeing.git
   cd interactive-chatbot-for-mental-wellbeing
   ```

2. **Set Up the Database**:
   - Install PostgreSQL and create a database (e.g., `mental_wellbeing_db`).
   - Update the `.env` file with your database URL:
     ```
     db_url=postgresql://username:password@localhost:5432/mental_wellbeing_db
     ```
   - The backend automatically creates tables on first run.

3. **Install Backend Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Frontend Dependencies**:
   ```bash
   npm install
   ```

5. **Run the Backend**:
   ```bash
   python server.py
   ```
   The Flask server runs on `http://127.0.0.1:5000` by default.

6. **Run the Frontend**:
   ```bash
   npm start
   ```
   Open `http://localhost:3000` (or your configured port) in your browser.

## Usage
- **Register/Login**: Create an account or log in to access the chatbot.
  ![Register page](screenshots/register_page.png)
- **Chat**: Type a message or select a predefined question to start a conversation.
- **Feedback/Rating**: Submit feedback or rate the chatbot via the sidebar options.
- Use tools like Postman to test API endpoints (e.g., `/post_userinput`, `/submit_feedback`).

## Model Training
The GPT-2 model was fine-tuned using six mental health datasets from Hugging Face:
- `marmikpandya/mental-health`
- `fadodr/mental_health_therapy`
- `Amod/mental_health_counseling_conversations`
- `jkhedri/psychology-dataset`
- `samhog/psychology-6k`
- `RAJJ18/mental_health_dataset` (subset of 3000 entries)

Training details:
- **Epochs**: 10
- **Batch Size**: 4
- **Learning Rate**: 3e-5
- **Evaluation**: BLEU score for response quality
- See `model.py` for full implementation.

## Issues Encountered
- **Database Setup**: Ensuring dynamic table creation and foreign key constraints worked seamlessly across environments.
- **Model Fine-Tuning**: Balancing GPU memory usage and training time with large datasets.
- **Real-Time Latency**: Simulating typing delays (1500ms) to mimic human-like responses while keeping the UI responsive.
- **Security**: Plain-text passwords in the database (to be addressed in future iterations).

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please report bugs or suggestions in the Issues section.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- Thanks to our team for their dedication and collaboration.
- Gratitude to Hugging Face for providing open-source mental health datasets.
- Appreciation for the open-source community behind React, Flask, and Transformers.

## Future Enhancements
- **Sentiment Analysis**: Integrate voice or text sentiment analysis for more adaptive responses.
- **Multi-Platform Support**: Develop a mobile app using React Native.
- **Expanded Datasets**: Include more diverse mental health data for broader coverage.
- **Security Upgrades**: Implement password hashing and HTTPS for production.

## Project Architecture
- **Frontend**: React/TypeScript SPA for user interaction.
- **Backend**: Flask REST API handling requests and model inference.
- **Database**: PostgreSQL for structured data storage.
- **ML Layer**: Fine-tuned GPT-2 for response generation.

For more details, explore the repository or contact the team!

