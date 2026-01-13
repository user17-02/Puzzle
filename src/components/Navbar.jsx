import { Link, NavLink } from "react-router-dom";
import "./Navbar.css";

export default function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg custom-navbar">
      <div className="container">
        {/* BRAND */}
        <Link className="navbar-brand fw-bold d-flex align-items-center" to="/">
       <img
  src="/puzzle world.png"
  alt="Puzzle World"
  className="navbar-logo"
/>

        </Link>

        {/* TOGGLER */}
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navMenu"
        >
          <span className="navbar-toggler-icon" />
        </button>

        {/* MENU */}
        <div className="collapse navbar-collapse" id="navMenu">
          <ul className="navbar-nav ms-auto mb-2 mb-lg-0 gap-lg-2">
            <li className="nav-item">
              <NavLink className="nav-link" to="/">
                Home
              </NavLink>
            </li>

            <li className="nav-item">
              <NavLink className="nav-link" to="/generator">
                Generator
              </NavLink>
            </li>
          </ul>

          {/* CTA */}
          <Link to="/generator" className="btn nav-cta ms-lg-3">
            Get Started
          </Link>
        </div>
      </div>
    </nav>
  );
}
