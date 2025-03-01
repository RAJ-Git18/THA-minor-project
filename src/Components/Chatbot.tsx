import React, { useState, useRef } from "react";
import { FaUserCircle, FaStar } from "react-icons/fa";
import axios from "axios";
import { useLocation, useNavigate } from "react-router-dom";

const Chatbot: React.FC = () => {
  const iconSize = 32;
  const [conversation, setConversation] = useState<
    { sender: string; text: string }[]
  >([]);
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [modal, setModal] = useState<"feedback" | "rating" | null>(null);
  const [rating, setRating] = useState<number>(0);
  const [feedback, setFeedback] = useState<string>("");
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState<boolean>(false);
  const location = useLocation();
  const navigate = useNavigate();

  const user_id = (location.state as { user_id?: string } | null)?.user_id;
  const user_name =
    (location.state as { user_name?: string } | null)?.user_name || "User";

  const questions: string[] = [
    "I'm feeling very sleepy day by day. What can I do?",
    "How can I establish a consistent sleep schedule?",
    "I'm feeling down and don't see the point in anything. Does therapy help?",
  ];

  const userInput = useRef<HTMLInputElement | null>(null);

  const handleSend = async (): Promise<void> => {
    if (userInput.current?.value) {
      const userInputText = userInput.current.value;
      setConversation((prev) => [
        ...prev,
        { sender: "user", text: userInputText },
      ]);
      userInput.current.value = "";
      setIsTyping(true);

      try {
        const response = await axios.post<{ chatbot_response: string }>(
          "http://127.0.0.1:5000/post_userinput",
          { input: userInputText, user_id }
        );
        setTimeout(() => {
          setConversation((prev) => [
            ...prev,
            { sender: "bot", text: response.data.chatbot_response },
          ]);
          setIsTyping(false);
        }, 1500);
      } catch (error) {
        console.error("Chatbot response error:", error);
        setIsTyping(false);
      }
    }
  };

  const handleFeedbackSubmit = async (): Promise<void> => {
    if (!feedback.trim())
      return alert("Please enter feedback before submitting.");

    try {
      await axios.post("http://127.0.0.1:5000/submit_feedback", {
        feedback,
        user_id,
      });
      alert("Thank you for your feedback!");
      setFeedback("");
      setModal(null);
    } catch (error) {
      console.error("Feedback submission error:", error);
    }
  };

  const handleRatingSubmit = async (rate: number): Promise<void> => {
    setRating(rate);
    setModal(null);

    try {
      await axios.post("http://127.0.0.1:5000/submit_rating", {
        rating: rate,
        user_id,
      });
      alert("Thank you for your rating!");
    } catch (error) {
      console.error("Rating submission error:", error);
    }
  };

  const handleProfileClick = (): void => {
    setIsProfileMenuOpen((prev) => !prev); // Toggle dropdown
  };

  const handleLogout = (): void => {
    setConversation([]); // Reset chat history
    setIsProfileMenuOpen(false);
    navigate("/"); // Redirect to initial page
  };

  const handleCloseMenu = (): void => {
    setIsProfileMenuOpen(false); // Just close the dropdown
  };

  return (
    <div className="bg-gray-200 h-screen p-4 flex gap-2 w-full">
      <div className="w-1/4 bg-gray-400 rounded-md p-4 flex flex-col">
        <h2 className="text-2xl font-semibold mb-4">Frequent Questions</h2>
        <div className="space-y-2">
          {questions.map((question, index) => (
            <button
              key={index}
              onClick={() =>
                userInput.current && (userInput.current.value = question)
              }
              className="w-full bg-gray-300 rounded p-2 text-left"
            >
              {question}
            </button>
          ))}
        </div>
        <button
          onClick={() => setModal("feedback")}
          className="mt-auto p-2 bg-gray-300 rounded"
        >
          💬 Submit Feedback
        </button>
        <button
          onClick={() => setModal("rating")}
          className="mt-2 p-2 justify-center bg-gray-300 rounded flex items-center gap-2"
        >
          <FaStar /> Rate Us
        </button>
      </div>

      <div className="w-3/4 bg-gray-300 rounded-md p-4 flex flex-col relative">
        <div className="flex justify-end">
          <button onClick={handleProfileClick} className="focus:outline-none">
            <FaUserCircle
              size={iconSize}
              className="cursor-pointer hover:text-gray-500"
            />
          </button>
          {isProfileMenuOpen && (
            <div className="absolute top-10 right-0 bg-white shadow-lg rounded-md p-2 w-40 z-10">
              <button
                onClick={handleCloseMenu}
                className="w-full text-left p-2 hover:bg-gray-100 rounded flex items-center justify-between"
              >
                Close <span className="text-gray-500">✕</span>
              </button>
              <button
                onClick={handleLogout}
                className="w-full text-left p-2 hover:bg-gray-100 rounded text-red-600"
              >
                Logout
              </button>
            </div>
          )}
        </div>
        <div className="flex-grow space-y-4 overflow-y-auto p-4">
          <div className="p-2 bg-gray-50 rounded-md w-1/2">
            Hi {user_name}! How can I help you?
          </div>
          {conversation.map((msg, index) => (
            <div
              key={index}
              className={`p-2 rounded-md min-h-10 w-1/2 ${
                msg.sender === "user" ? "bg-gray-400 ml-auto" : "bg-gray-50"
              }`}
            >
              {msg.text}
            </div>
          ))}
          {isTyping && (
            <div className="p-2 bg-gray-50 rounded-md w-1/2">
              Bot is typing...
            </div>
          )}
        </div>
        <div className="flex items-center mt-4">
          <input
            ref={userInput}
            type="text"
            placeholder="Message ChatBot"
            className="flex-grow p-2 rounded-l-md border"
          />
          <button onClick={handleSend} className="p-2 bg-gray-400 rounded-r-md">
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
              <path d="M2.003 21L23 12 2.003 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </div>

      {modal === "feedback" && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-4 rounded-md text-center w-96">
            <h2 className="text-lg font-semibold mb-2">Submit Feedback</h2>
            <textarea
              className="w-full p-2 border rounded"
              rows={4}
              placeholder="Write your feedback..."
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
            />
            <div className="flex justify-between mt-2">
              <button
                onClick={handleFeedbackSubmit}
                className="p-2 bg-gray-300 rounded"
              >
                Submit
              </button>
              <button
                onClick={() => setModal(null)}
                className="p-2 bg-gray-300 rounded"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      {modal === "rating" && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-4 rounded-md text-center max-w-sm w-full">
            <h2 className="text-lg font-semibold mb-2">Rate Us</h2>
            <div className="flex justify-center space-x-2 mb-4">
              {[1, 2, 3, 4, 5].map((star) => (
                <FaStar
                  key={star}
                  size={24}
                  className={`cursor-pointer ${
                    star <= rating ? "text-yellow-500" : "text-gray-400"
                  }`}
                  onClick={() => handleRatingSubmit(star)}
                />
              ))}
            </div>
            <button
              onClick={() => setModal(null)}
              className="mt-2 p-2 bg-gray-300 rounded w-full"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
