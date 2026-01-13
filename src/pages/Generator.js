import { useState, useEffect } from "react";
import { FaMagic, FaDownload } from "react-icons/fa";
import "./Generator.css";

const puzzleImages = {
  sudoku: "/sudoku.jpg",
  crossword: "/crossword.jpg",
  maze: "/Maze.jpg",
  wordsearch: "/wordsearch.jpg",
};

export default function Generator() {
  const [puzzleType, setPuzzleType] = useState("sudoku");
  const [difficulty, setDifficulty] = useState("easy");
  const [count, setCount] = useState(25);
  const [loading, setLoading] = useState(false);
  const [pdfUrl, setPdfUrl] = useState(null);

  // Cleanup blob URL when component unmounts or regenerates
  useEffect(() => {
    return () => {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    };
  }, [pdfUrl]);

  const handleGenerate = async () => {
    setLoading(true);
    if (pdfUrl) {
      URL.revokeObjectURL(pdfUrl);
      setPdfUrl(null);
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          puzzle_type: puzzleType,
          difficulty: difficulty,
          count: count,
        }),
      });

      if (!response.ok) {
        throw new Error("Generation failed");
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setPdfUrl(url);
    } catch (err) {
      console.error(err);
      alert("Failed to generate puzzle book. Check backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!pdfUrl) return;

    const a = document.createElement("a");
    a.href = pdfUrl;
    a.download = `${puzzleType}_puzzle_book.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div
      className="generator-page"
      style={{
        backgroundImage: "url(/bg.jpg)",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        minHeight: "100vh",
      }}
    >
      <div className="generator-overlay">
        <div className="container py-5">
          <div className="row align-items-start g-5">

            {/* LEFT PANEL */}
            <div className="col-lg-5">
              <div className="generator-card">
                <h2 className="fw-bold mb-2">Puzzle Generator</h2>
                <p className="text-muted-light mb-4">
                  Create KDP-ready puzzle books using AI.
                </p>

                <label className="form-label">Puzzle Type</label>
                <select
                  className="form-select mb-3"
                  value={puzzleType}
                  onChange={(e) => setPuzzleType(e.target.value)}
                >
                  <option value="sudoku">Sudoku</option>
                  <option value="crossword">Crossword</option>
                  <option value="maze">Maze</option>
                  <option value="wordsearch">Word Search</option>
                </select>

                <label className="form-label">Difficulty</label>
                <select
                  className="form-select mb-3"
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>

                <label className="form-label">Number of Puzzles</label>
                <input
                  type="number"
                  className="form-control mb-4"
                  min={1}
                  max={100}
                  value={count}
                  onChange={(e) => setCount(Number(e.target.value))}
                />

                <button
                  className="btn btn-generate w-100 mb-3"
                  onClick={handleGenerate}
                  disabled={loading}
                >
                  <FaMagic className="me-2" />
                  {loading ? "Generating…" : "Generate Puzzle Book"}
                </button>

                {pdfUrl && (
                  <button
                    className="btn btn-outline-light w-100"
                    onClick={handleDownload}
                  >
                    <FaDownload className="me-2" />
                    Download PDF
                  </button>
                )}
              </div>
            </div>

            {/* RIGHT PANEL */}
            <div className="col-lg-7">
              {!pdfUrl ? (
                <div className="preview-card text-center">
                  <h5 className="fw-bold mb-3 text-uppercase">
                    {puzzleType} Reference Layout
                  </h5>
                  <img
                    src={puzzleImages[puzzleType]}
                    alt={puzzleType}
                    className="img-fluid preview-image"
                  />
                  <p className="text-muted-light mt-3">
                    Example layout preview before generation
                  </p>
                </div>
              ) : (
                <div className="preview-card">
                  <h5 className="fw-bold mb-3 text-uppercase">
                    Live PDF Preview
                  </h5>
                  <iframe
                    src={pdfUrl}
                    title="Puzzle Book Preview"
                    width="100%"
                    height="650px"
                    style={{
                      borderRadius: "12px",
                      border: "1px solid rgba(255,255,255,0.2)",
                      background: "#000",
                    }}
                  />
                </div>
              )}
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
