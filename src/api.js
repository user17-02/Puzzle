const API = "http://localhost:8000";

const res = await fetch(`${API}/generate`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    puzzle_type: puzzleType,
    difficulty,
    count,
    title,
  }),
});
