import { useEffect, useState } from "react";
import { ProgressBar, Card } from "react-bootstrap";
import { api } from "../api/api";

export default function ProgressTracker({ jobId }) {
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      const res = await api.get(`/progress/${jobId}`);
      setProgress(res.data);
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId]);

  if (!progress) return null;

  return (
    <Card className="p-3 mt-4">
      <strong>{progress.step}</strong>
      <ProgressBar
        now={progress.percent}
        label={`${progress.percent}%`}
        className="mt-2"
      />
    </Card>
  );
}
