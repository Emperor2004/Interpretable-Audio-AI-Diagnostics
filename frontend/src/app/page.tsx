'use client'; // This marks the component as a Client Component

import { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';

// --- Define the structure of the API response ---
interface Prediction {
  label: string;
  confidence: number;
}

interface AnalysisResult {
  top_prediction: Prediction;
  all_predictions: Prediction[];
  xai_heatmap_image: string; // This will be a base64 data URL
  xai_explanation: string;
}

// --- Main Page Component ---
export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setResult(null); // Clear previous results
      setError(null);
    }
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an audio file first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Send the file to the FastAPI backend
      const response = await axios.post<AnalysisResult>(
        'http://localhost:8000/analyze_audio',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setResult(response.data);
    } catch (err: any) {
      console.error(err);
      setError(
        'An error occurred during analysis: ' +
          (err.response?.data?.detail || err.message)
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-start bg-gray-100 p-8">
      <main className="w-full max-w-4xl rounded-lg bg-white p-8 shadow-xl">
        {/* --- Header --- */}
        <div className="mb-6 border-b border-gray-200 pb-4">
          <h1 className="text-center text-4xl font-bold text-gray-800">
            Interpretable Audio AI Diagnostics
          </h1>
          <p className="mt-2 text-center text-lg text-gray-600">
            Upload an audio file (.wav, .mp3) to classify its content and
            understand *why* the model made its decision.
          </p>
        </div>

        {/* --- File Upload Form --- */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex flex-col rounded-md border border-dashed border-gray-400 p-6">
            <label
              htmlFor="file-upload"
              className="mb-2 cursor-pointer text-sm font-medium text-gray-700"
            >
              Select Audio File:
            </label>
            <input
              id="file-upload"
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="mb-2 block w-full text-sm text-gray-500
                         file:mr-4 file:rounded-full file:border-0
                         file:bg-blue-50 file:px-4
                         file:py-2 file:text-sm
                         file:font-semibold file:text-blue-700
                         hover:file:bg-blue-100"
            />
            <button
              type="submit"
              disabled={!file || isLoading}
              className="mt-4 w-full rounded-md bg-blue-600 px-4 py-2 text-lg font-semibold text-white
                         shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed
                         disabled:opacity-50"
            >
              {isLoading ? 'Analyzing...' : 'Analyze Audio'}
            </button>
          </div>
        </form>

        {/* --- Loading Spinner --- */}
        {isLoading && (
          <div className="flex justify-center">
            <div
              className="h-12 w-12 animate-spin rounded-full border-4 border-t-4 border-gray-200 border-t-blue-600"
              role="status"
            >
              <span className="sr-only">Loading...</span>
            </div>
          </div>
        )}

        {/* --- Error Message --- */}
        {error && (
          <div
            className="rounded-md border border-red-400 bg-red-100 p-4 text-red-700"
            role="alert"
          >
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        {/* --- Results Display --- */}
        {result && (
          <section className="animate-fadeIn rounded-lg border border-gray-200 bg-gray-50 p-6">
            <h2 className="mb-4 border-b border-gray-300 pb-2 text-3xl font-semibold text-gray-800">
              Analysis Results
            </h2>

            {/* --- Top Prediction --- */}
            <div className="mb-6 rounded-lg bg-white p-4 text-center shadow">
              <span className="text-lg text-gray-600">Top Prediction:</span>
              <h3 className="text-5xl font-bold text-blue-700">
                {result.top_prediction.label}
              </h3>
              <p className="text-2xl font-light text-gray-500">
                ({result.top_prediction.confidence}% Confidence)
              </p>
            </div>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              {/* --- XAI Heatmap --- */}
              <div className="rounded-lg bg-white p-4 shadow">
                <h4 className="mb-2 text-xl font-semibold text-gray-700">
                  XAI Attention Heatmap
                </h4>
                <p className="mb-4 text-sm text-gray-600">
                  This image shows *where* the model &quot;listened&quot; to make
                  its decision. Red areas indicate high importance.
                </p>
                <img
                  src={result.xai_heatmap_image}
                  alt="XAI Attention Heatmap"
                  className="w-full rounded-md border border-gray-300"
                />
              </div>

              {/* --- XAI Explanation --- */}
              <div className="rounded-lg bg-white p-4 shadow">
                <h4 className="mb-2 text-xl font-semibold text-gray-700">
                  Plain-Language Explanation
                </h4>
                <div
                  className="prose prose-blue"
                  dangerouslySetInnerHTML={{ __html: result.xai_explanation }}
                />
                
                {/* --- Top 5 Predictions --- */}
                <h4 className="mb-2 mt-6 text-xl font-semibold text-gray-700">
                  Other Possibilities
                </h4>
                <ul className="list-inside list-disc space-y-1">
                  {result.all_predictions.slice(1).map((pred) => (
                    <li key={pred.label} className="text-gray-600">
                      <strong>{pred.label}:</strong> {pred.confidence}%
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}