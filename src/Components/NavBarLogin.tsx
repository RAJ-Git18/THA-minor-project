import React from "react";
import { Link } from "react-router-dom";

/**
 * NavBar Component
 *
 * This component renders a navigation bar with a brand logo and a login button.
 * It uses React Router's Link component for navigation.
 */
const NavBar = () => {
  return (
    <div className="flex justify-between bg-gradient-to-r from-gray-700 to-gray-900 p-3">
      {/* Logo Section */}
      <div>
        <Link
          to="/"
          className="text-white text-4xl font-bold hover:text-teal-300 transition duration-300 ease-in-out"
        >
          Mental<span className="text-blue-500">Matters</span>
        </Link>
      </div>

      {/* Login Button */}
      <div>
        <Link
          to="/login"
          className="px-6 py-3 bg-blue-500 text-white rounded-md block text-center"
        >
          LOGIN
        </Link>
      </div>
    </div>
  );
};

export default NavBar;
