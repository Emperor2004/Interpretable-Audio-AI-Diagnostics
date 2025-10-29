'use client'; // This marks the component as a Client Component

import { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';

// --- Define the structure of the API response ---
interface Prediction {
  label: string;
  confidence: number;
}

interface ExplanationItem {
  label: string;
  count: number;
}
interface ExplanationResult {
  primary_symptoms: ExplanationItem[];
  other_sounds: ExplanationItem[];
}

interface AnalysisResult {
  top_prediction: Prediction;
  all_predictions: Prediction[];
  xai_detection_plot: string;   // <-- This is the dual-plot
  xai_attention_heatmap: string; // <-- This is the Grad-CAM plot
  xai_explanation: ExplanationResult;
}

// --- Main Page Component ---
export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (audioURL) {
      URL.revokeObjectURL(audioURL);
    }
    if (e.target.files) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setAudioURL(URL.createObjectURL(selectedFile));
      setResult(null);
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

    // Corrected Typo: FormData instead of formData
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post<AnalysisResult>(
        'http://localhost:8000/analyze_audio',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
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
      {/* Corrected Closing Tag: Ensure <main> is properly closed at the end */}
      <main className="w-full max-w-4xl rounded-lg bg-white p-8 shadow-xl">
        {/* --- Header --- */}
        <div className="mb-6 border-b border-gray-200 pb-4">
          <h1 className="text-center text-4xl font-bold text-gray-800">
            Interpretable Audio AI Diagnostics
          </h1>
          <p className="mt-2 text-center text-lg text-gray-600">
            Upload an audio file to demonstrate ResNet50 + Grad-CAM.
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
            {/* Corrected Syntax: Removed extra characters/spaces in className */}
            <button
              type="submit"
              disabled={!file || isLoading}
              className="mt-4 w-full rounded-md bg-blue-600 px-4 py-2 text-lg font-semibold text-white shadow-sm hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? 'Analyzing...' : 'Analyze Audio'}
            </button>
          </div>
        </form>

        {/* --- Loading Spinner / Error --- */}
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
        {/* Corrected: Ensure result is checked before accessing its properties */}
        {result && (
          <section className="animate-fadeIn rounded-lg border border-gray-200 bg-gray-50 p-6 grid grid-cols-1 gap-6">

            {/* --- Audio Player --- */}
            {audioURL && (
              <div className="rounded-lg bg-white p-4 shadow">
                <h4 className="mb-2 text-xl font-semibold text-gray-700">
                  Playback Audio
                </h4>
                <audio controls src={audioURL} className="w-full" />
              </div>
            )}

            {/* --- Full Audio Plot --- */}
            <div className="rounded-lg bg-white p-4 shadow">
              <h4 className="mb-2 text-xl font-semibold text-gray-700">
                Full Audio Plot
              </h4>
              <p className="mb-4 text-sm text-gray-600">
                This plot shows the audio waveform and full spectrogram.
              </p>
              <div className="flex w-full justify-center">
                <img
                  src={result.xai_detection_plot}
                  alt="Symptom Detection Plot"
                  className="max-w-full rounded-md border border-gray-300 w-full"
                />
              </div>
            </div>

            {/* --- XAI Grad-CAM Heatmap --- */}
            {result.xai_attention_heatmap && (
              <div className="rounded-lg bg-white p-4 shadow">
                <h4 className="mb-2 text-xl font-semibold text-gray-700">
                  XAI (Grad-CAM) Heatmap
                </h4>
                <p className="mb-4 text-sm text-gray-600">
                  This is the "AI brain scan" for the first non-silent chunk.
                  Red areas on the spectrogram show what the model
                  "listened" to the most to make its decision.
                </p>
                <div className="flex w-full justify-center">
                  <img
                    src={result.xai_attention_heatmap}
                    alt="XAI Grad-CAM Heatmap"
                    className="max-w-full rounded-md border border-gray-300 w-full"
                  />
                </div>
              </div>
            )}

            {/* --- XAI Explanation --- */}
            <div className="rounded-lg bg-white p-4 shadow">
              <h4 className="mb-2 text-xl font-semibold text-gray-700">
                Analysis Explanation
              </h4>
              <div className="space-y-4 text-gray-700">

                {/* --- Primary Symptoms --- */}
                {/* Corrected Syntax: Ensure proper mapping and closing tags */}
                {result.xai_explanation.primary_symptoms.length > 0 ? (
                  <div>
                    <h5 className="font-semibold">Primary Symptoms Detected:</h5>
                    <ul className="list-inside list-disc pl-2">
                      {result.xai_explanation.primary_symptoms.map((item) => (
                        <li key={item.label}>
                          <strong>{item.count} {item.label} event(s)</strong>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {/* --- Other Sounds --- */}
                {/* Corrected Syntax: Ensure proper mapping and closing tags */}
                {result.xai_explanation.other_sounds.length > 0 ? (
                  <div>
                    <h5 className="font-semibold">Other Sounds Detected:</h5>
                    <ul className="list-inside list-disc pl-2">
                      {result.xai_explanation.other_sounds.map((item) => (
                        <li key={item.label}>
                          {item.label} (detected in {item.count} analysis chunks)
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {/* --- No Detections Case --- */}
                {result.xai_explanation.primary_symptoms.length === 0 &&
                  result.xai_explanation.other_sounds.length === 0 && (
                    <p>
                      <strong>No significant audio events were detected.</strong> The model
                      did not find any target symptoms or other identifiable
                      sounds with high confidence.
                    </p>
                  )}
              </div> {/* End space-y-4 div */}
            </div> {/* End Explanation Card div */}

          </section> /* End Results Section */
        )} {/* End Conditional Render for result */}
      </main> {/* Corrected: Added closing tag for main */}
    </div> /* End Root div */
  );
}