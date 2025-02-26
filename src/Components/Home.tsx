import { Link } from "react-router-dom"; // Import Link for navigation (though not used in this file)
import Register_Page from "./Register"; // Import the Register component
import NavBar from "./NavBarLogin"; // Import the NavBar component

// Define the Home component
const Home = () => {
  return (
    <>
      {/* Render the navigation bar at the top of the page */}
      <div>
        <NavBar />
      </div>

      {/* Main content area */}
      <div className="flex flex-col items-center min-h-screen bg-gray-100 w-100vh">
        <div className="space-y-4">
          {/* Render the Register_Page component */}
          <Register_Page />
        </div>
      </div>
    </>
  );
};

// Export the Home component as the default export
export default Home;
