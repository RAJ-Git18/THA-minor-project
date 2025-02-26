import React, { useState } from "react"; // Import React and useState hook
import { EyeIcon } from "@heroicons/react/24/outline"; // Import EyeIcon for password visibility toggle
import { useNavigate } from "react-router-dom"; // Import useNavigate for programmatic navigation
import { Link } from "react-router-dom"; // Import Link for navigation
import axios from "axios"; // Import axios for making HTTP requests
import NavBar from "./NavBarRegister"; // Import the NavBar component

// Define the Login component
const Login = () => {
  const navigate = useNavigate(); // Hook to programmatically navigate

  // Define the User interface for type safety
  interface User {
    email: string;
    password: string;
  }

  // State to manage user login details
  const [userDetails, setUserDetails] = useState<User>({
    email: "",
    password: "",
  });

  // State to track form submission status
  const [Submit, setSubmit] = useState<boolean>(false);

  // State to store form validation errors
  const [formErrors, setFormErrors] = useState<any>({});

  // State to toggle password visibility
  const [showPassword, setShowPassword] = useState<boolean>(false);

  // State to toggle the visibility of the eye icon for password
  const [showEye, setShowEye] = useState<boolean>(false);

  // Function to handle input changes
  function handleChange(event: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = event.target;

    // Show/hide the eye icon based on whether the password field has a value
    if (name === "password") {
      if (value !== "") {
        setShowEye(true);
      } else {
        setShowEye(false);
      }
    }

    // Update the userDetails state with the new input value
    setUserDetails({
      ...userDetails,
      [name]: value,
    });
  }

  // Function to handle login form submission
  async function handleLogin(event: any) {
    event.preventDefault(); // Prevent default form submission behavior

    // Validate the form and set errors if any
    setFormErrors(validateForm(userDetails));

    // If there are validation errors, stop the submission
    if (Object.values(formErrors).some((error) => error !== "")) {
      console.log(formErrors);
      setSubmit(false);
      return;
    }

    try {
      // Send a POST request to the backend to log in the user
      const post_response = await axios.post(
        "http://127.0.0.1:5000/post_login_data",
        userDetails
      );
      console.log("Response:", post_response.data);
      console.log("The user who logged in was:", post_response.data.user);

      // If login is successful, reset the form and navigate to the chatbot interface
      if (post_response.data.message === "Login successful") {
        setSubmit(true);
        setUserDetails({
          email: "",
          password: "",
        });

        // Pass user_id and user_name to the next page using navigation state
        const userData = post_response.data.user;
        navigate("/chatbot_interface", {
          state: { user_id: userData.id, user_name: userData.name },
        });
      } else {
        // If login fails, show an alert with the error message
        setSubmit(false);
        alert(post_response.data.message);
      }
    } catch (error) {
      // Handle errors during login
      console.error("Could not log in the user");
      alert("The username doesn't exist.");
      setSubmit(false);
    }
  }

  // Function to validate the login form
  function validateForm(user: User) {
    const error: User = {
      email: "",
      password: "",
    };
    const regex = /^[^\s+@]+@[^\s@]+\.[^\s@]{2,}$/i; // Regex for email validation

    // Validate email
    if (!user.email) {
      error.email = "Email is required";
    } else if (!regex.test(user.email)) {
      error.email = "Please enter a valid email address";
    }

    // Validate password
    if (!user.password) {
      error.password = "Password is required";
    }

    return error;
  }

  // Function to show the password when the eye icon is pressed
  function handleMouseDownPassword() {
    setShowPassword(true);
  }

  // Function to hide the password when the eye icon is released
  function handleMouseUpPassword() {
    setShowPassword(false);
  }

  return (
    <div>
      {/* Render the navigation bar */}
      <div>
        <NavBar />
      </div>

      {/* Main login form */}
      <div className="flex items-center h-screen bg-gradient-to-r bg-gray-100">
        <form className="bg-white p-6 rounded-xl shadow-lg w-full max-w-sm mx-auto space-y-4">
          {/* Form heading */}
          <h1 className="text-xl font-semibold text-center text-gray-800 mb-4">
            Let’s Get You Logged In
          </h1>

          {/* Input fields */}
          <div className="space-y-3">
            {/* Email input */}
            <input
              type="email"
              name="email"
              id="email"
              value={userDetails.email}
              onChange={handleChange}
              placeholder="Email address"
              className="w-full px-3 py-2 border-2 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200 hover:border-teal-400"
            />

            {/* Password input with eye icon for visibility toggle */}
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                id="password"
                value={userDetails.password}
                onChange={handleChange}
                placeholder="Password"
                className="w-full px-3 py-2 border-2 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200 hover:border-teal-400"
              />
              <button
                type="button"
                onMouseDown={handleMouseDownPassword}
                onMouseUp={handleMouseUpPassword}
                onMouseLeave={handleMouseUpPassword}
                className="absolute top-1/2 right-4 transform -translate-y-1/2 text-gray-500 hover:text-teal-500"
              >
                {showEye ? <EyeIcon className="h-4 w-4" /> : <></>}
              </button>
            </div>
          </div>

          {/* Login button */}
          <div className="mt-4">
            <button
              type="button"
              onClick={handleLogin}
              className="w-full py-2 bg-teal-500 text-white font-semibold rounded-md hover:bg-teal-600 focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200"
            >
              Login
            </button>
          </div>

          {/* Link to the registration page */}
          <div className="text-center mt-3">
            <p className="text-gray-600 text-sm">
              Don't have an account?{" "}
              <Link
                className="text-teal-500 hover:text-teal-700 font-semibold"
                to="/"
              >
                Register Now
              </Link>
            </p>
          </div>
        </form>
      </div>
    </div>
  );
};

// Export the Login component as the default export
export default Login;
