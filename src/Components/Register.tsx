import React, { useState } from "react"; // Import React and useState hook
import { EyeIcon } from "@heroicons/react/24/outline"; // Import EyeIcon for password visibility toggle
import { Link } from "react-router-dom"; // Import Link for navigation
import axios from "axios"; // Import axios for making HTTP requests

// Define the Register_Page component
const Register_Page = () => {
  // Define the User interface for type safety
  interface User {
    firstname: string;
    lastname: string;
    email: string;
    password: string;
    confirmPassword: string;
  }

  // State to manage user registration details
  const [userDetails, setUserDetails] = useState<User>({
    firstname: "",
    lastname: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  // State to track form submission status
  const [submit, setSubmit] = useState<boolean>(false);

  // State to toggle password visibility
  const [showPassword, setShowPassword] = useState<boolean>(false);

  // State to toggle confirm password visibility
  const [showConfirmPassword, setShowConfirmPassword] =
    useState<boolean>(false);

  // State to toggle the visibility of the eye icon for password
  const [showEyePass, setShowEyePass] = useState<boolean>(false);

  // State to toggle the visibility of the eye icon for confirm password
  const [showEyePassConf, setShowEyePassConf] = useState<boolean>(false);

  // Function to handle input changes
  function handleChange(event: React.ChangeEvent<HTMLInputElement>) {
    const { name, value } = event.target;

    // Show/hide the eye icon based on whether the password or confirm password field has a value
    if (name === "password") {
      if (value !== "") {
        setShowEyePass(true);
      } else {
        setShowEyePass(false);
      }
    } else if (name === "confirmPassword") {
      if (value !== "") {
        setShowEyePassConf(true);
      } else {
        setShowEyePassConf(false);
      }
    }

    // Update the userDetails state with the new input value
    setUserDetails((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  }

  // Function to validate the registration form
  function validateForm(user: User) {
    const error: User = {
      firstname: "",
      lastname: "",
      email: "",
      password: "",
      confirmPassword: "",
    };
    const regex = /^[^\s+@]+@[^\s@]+\.[^\s@]{2,}$/i; // Regex for email validation

    // Validate first name
    if (!user.firstname) {
      error.firstname = "First Name is required";
    }

    // Validate last name
    if (!user.lastname) {
      error.lastname = "Last Name is required";
    }

    // Validate email
    if (!user.email) {
      error.email = "Email is required";
    } else if (!regex.test(user.email)) {
      error.email = "This is not a valid email format!";
    }

    // Validate password
    if (!user.password) {
      error.password = "Password is required";
    } else if (user.password.length < 4) {
      error.password = "Password must be more than 4 characters";
    } else if (user.password.length > 10) {
      error.password = "Password cannot exceed more than 10 characters";
    }

    // Validate confirm password
    if (!user.confirmPassword) {
      error.confirmPassword = "Confirm Password is required";
    } else if (user.confirmPassword !== user.password) {
      error.confirmPassword = "Confirm password and password should be same";
    }

    return error;
  }

  // Function to handle form submission
  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault(); // Prevent default form submission behavior

    // Validate the form and set errors if any
    let errors = validateForm(userDetails);

    // If there are validation errors, stop the submission
    if (Object.values(errors).some((error) => error !== "")) {
      console.log(errors);
      setSubmit(false);
      return;
    }

    try {
      // Send a POST request to the backend to register the user
      const response = await axios.post("http://127.0.0.1:5000", userDetails, {
        headers: {
          "Content-Type": "application/json", // Set content type to JSON
        },
      });
      console.log("Response from backend:", response.data);

      // Show success message and reset the form
      alert("Your account is registered \n Now you can login");
      setSubmit(true);
      setUserDetails({
        firstname: "",
        lastname: "",
        email: "",
        password: "",
        confirmPassword: "",
      });
    } catch (error) {
      // Handle errors during registration
      console.error("Could not register the user", error);
      setSubmit(false);
    }
  }

  // Function to show the password when the eye icon is pressed
  const handleMouseDownPassword = () => {
    setShowPassword(true);
  };

  // Function to hide the password when the eye icon is released
  const handleMouseUpPassword = () => {
    setShowPassword(false);
  };

  // Function to show the confirm password when the eye icon is pressed
  const handleMouseDownConfPassword = () => {
    setShowConfirmPassword(true);
  };

  // Function to hide the confirm password when the eye icon is released
  const handleMouseUpConfPassword = () => {
    setShowConfirmPassword(false);
  };

  return (
    <>
      {/* Main registration form container */}
      <div className="mt-7 bg-gradient-to-r from-blue-200 to-teal-200 px-4 p-5 w-full rounded-lg min-w-full">
        <form
          onSubmit={handleSubmit}
          className="bg-white p-6 rounded-xl shadow-lg w-full max-w-sm mx-auto space-y-4"
        >
          {/* Form heading */}
          <h3 className="text-center text-xl font-semibold text-to-teal-800 mb-4">
            Welcome To Mental Matters
          </h3>
          <h3 className="text-center text-base text-gray-600 mb-3">
            Create Your Account
          </h3>

          {/* Input fields */}
          <div className="space-y-3">
            {/* First Name input */}
            <input
              type="text"
              name="firstname"
              id="firstname"
              value={userDetails.firstname}
              onChange={handleChange}
              placeholder="First Name"
              className="w-full px-3 py-2 border-2 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200 hover:border-teal-400"
            />

            {/* Last Name input */}
            <input
              type="text"
              name="lastname"
              id="lastname"
              value={userDetails.lastname}
              onChange={handleChange}
              placeholder="Last Name"
              className="w-full px-3 py-2 border-2 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200 hover:border-teal-400"
            />

            {/* Email input */}
            <input
              type="email"
              name="email"
              id="email"
              value={userDetails.email}
              onChange={handleChange}
              placeholder="Email"
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
                {showEyePass ? <EyeIcon className="h-4 w-4" /> : <></>}
              </button>
            </div>

            {/* Confirm Password input with eye icon for visibility toggle */}
            <div className="relative">
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                id="confirmPassword"
                value={userDetails.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm Password"
                className="w-full px-3 py-2 border-2 border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200 hover:border-teal-400"
              />
              <button
                type="button"
                onMouseDown={handleMouseDownConfPassword}
                onMouseUp={handleMouseUpConfPassword}
                onMouseLeave={handleMouseUpConfPassword}
                className="absolute top-1/2 right-4 transform -translate-y-1/2 text-gray-500 hover:text-teal-500"
              >
                {showEyePassConf ? <EyeIcon className="h-4 w-4" /> : <></>}
              </button>
            </div>
          </div>

          {/* Register button */}
          <div className="mt-5">
            <button
              type="submit"
              className="w-full py-2 bg-teal-500 text-white font-semibold rounded-md hover:bg-teal-600 focus:outline-none focus:ring-2 focus:ring-teal-500 transition duration-200"
            >
              Register
            </button>
          </div>

          {/* Link to the login page */}
          <div className="text-center mt-3">
            <p className="text-gray-600 text-sm">
              Already Have An Account? <br />
              <Link
                to="/login"
                className="text-teal-500 hover:text-teal-700 font-semibold"
              >
                Login
              </Link>
            </p>
          </div>
        </form>
      </div>
    </>
  );
};

// Export the Register_Page component as the default export
export default Register_Page;
