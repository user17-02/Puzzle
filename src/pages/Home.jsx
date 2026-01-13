import { Link } from "react-router-dom";
import {
  FaRobot,
  FaPrint,
  FaFilePdf,
  FaBolt,
  FaStar,
  FaCheckCircle,
} from "react-icons/fa";
import "./Home.css";

export default function Home() {
  return (
    <>
      {/* HERO */}
      <section className="hero-section">
        <div className="container">
          <div className="row align-items-center g-5">
            {/* LEFT */}
            <div className="col-lg-6 animate-up">
              <h1 className="display-4 fw-bold mb-3">
                Create <span className="highlight">Print-Ready</span>
                <br />
                Puzzle Books with AI
              </h1>

              <p className="lead text-muted-light">
                Generate Crossword, Sudoku, Maze & Word Search books using
                CrewAI — fully compatible with Amazon KDP.
              </p>

              <div className="d-flex gap-3 mt-4">
                <Link to="/generator" className="btn btn-primary btn-lg">
                  Generate Now
                </Link>
                <a href="#pricing" className="btn btn-outline-light btn-lg">
                  Pricing
                </a>
              </div>
            </div>

            {/* RIGHT */}
            <div className="col-lg-6 text-center animate-scale">
              <div className="hero-circle">
                <img
                  src="/puzzle.png"
                  alt="Puzzle"
                  className="img-fluid"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* STATS */}
      <section className="stats-section">
        <div className="container">
          <div className="row text-center g-4">
            {[
              ["10K+", "Puzzles Generated"],
              ["1,200+", "Books Published"],
              ["4.9★", "Average Rating"],
              ["95%", "KDP Approval Rate"],
            ].map(([num, label], i) => (
              <div key={i} className="col-md-3 animate-up">
                <h2 className="fw-bold">{num}</h2>
                <p className="text-muted">{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section className="features-section">
        <div className="container text-center">
          <h2 className="fw-bold mb-5">Why Creators Love It</h2>

          <div className="row g-4">
            {[
              [FaRobot, "CrewAI Powered"],
              [FaPrint, "KDP Ready PDFs"],
              [FaFilePdf, "Live Preview"],
              [FaBolt, "Fast Generation"],
            ].map(([Icon, text], i) => (
              <div key={i} className="col-md-6 col-lg-3 animate-up">
                <div className="feature-card">
                  <Icon size={36} />
                  <h5 className="mt-3 fw-bold">{text}</h5>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* TESTIMONIALS */}
      <section className="testimonials-section">
        <div className="container">
          <h2 className="text-center fw-bold mb-5">
            Trusted by Puzzle Publishers
          </h2>

          <div className="row g-4">
            {[
              ["Sarah M.", "KDP Publisher", "Published 7 books in 3 weeks."],
              ["David R.", "Educator", "Saved hundreds of hours."],
              ["Anita K.", "Indie Author", "Zero KDP rejections."],
            ].map(([name, role, quote], i) => (
              <div key={i} className="col-md-4 animate-up">
                <div className="testimonial-card">
                  <FaStar className="star" />
                  <p className="mt-3">“{quote}”</p>
                  <h6 className="fw-bold mb-0">{name}</h6>
                  <small className="text-muted">{role}</small>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* PRICING */}
     {/* PRICING */}
<section id="pricing" className="pricing-section">
  <div className="container">
    <div className="text-center mb-5">
      <h2 className="fw-bold">Simple, Creator-Friendly Pricing</h2>
      <p className="text-muted-light mt-2">
        Start free. Upgrade only when you’re ready to publish at scale.
      </p>
    </div>

    <div className="row g-4 justify-content-center align-items-stretch">

      {/* FREE PLAN */}
      <div className="col-md-6 col-lg-4">
        <div className="pricing-card h-100">
          <span className="badge bg-secondary mb-3">Starter</span>

          <h4 className="fw-bold mt-2">Free</h4>
          <p className="pricing-price">$0</p>
          <p className="pricing-desc">
            Perfect for testing puzzle quality and layouts.
          </p>

          <ul className="pricing-list">
            <li><FaCheckCircle /> 10 puzzles total</li>
            <li><FaCheckCircle /> Single puzzle type</li>
            <li><FaCheckCircle /> PDF export</li>
            <li><FaCheckCircle /> Answer keys</li>
          </ul>

          <button className="btn btn-outline-light w-100 mt-auto">
            Get Started Free
          </button>
        </div>
      </div>

      {/* PRO PLAN */}
      <div className="col-md-6 col-lg-4">
        <div className="pricing-card featured h-100">
          <span className="badge badge-pro mb-3">Most Popular</span>

          <h4 className="fw-bold mt-2">Pro</h4>
          <p className="pricing-price">$19<span>/mo</span></p>
          <p className="pricing-desc">
            Built for serious KDP publishers and educators.
          </p>

          <ul className="pricing-list">
            <li><FaCheckCircle /> Unlimited puzzles</li>
            <li><FaCheckCircle /> All puzzle types</li>
            <li><FaCheckCircle /> KDP trim sizes</li>
            <li><FaCheckCircle /> Live PDF preview</li>
            <li><FaCheckCircle /> Faster AI generation</li>
          </ul>

          <Link to="/generator" className="btn btn-primary w-100 mt-auto">
            Upgrade to Pro
          </Link>
        </div>
      </div>

    </div>
  </div>
</section>

    </>
  );
}
