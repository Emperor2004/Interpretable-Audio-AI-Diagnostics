// frontend/src/app/page.tsx

'use client';

import { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';

// --- Define the structure of the API response ---
interface Prediction {
    index: number;
    label: string;
    probability: number;
}

interface AnalysisResult {
    predictions: Prediction[];
    cam_base64: string;
    prediction_time: number;
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
        <div className="flex min-h-screen flex-col items-center justify-start bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
            <main className="w-full max-w-5xl rounded-xl bg-white p-8 shadow-2xl">
                {/* --- Header --- */}
                <div className="mb-8 border-b border-gray-200 pb-6">
                    <h1 className="text-center text-4xl font-bold text-gray-900">
                        🎵 Audio Classification with XAI
                    </h1>
                    <p className="mt-3 text-center text-lg text-gray-600">
                        Upload an audio file to analyze it with PANNs CNN14 and visualize AI decisions using Grad-CAM
                    </p>
                </div>

                {/* --- File Upload Form --- */}
                <form onSubmit={handleSubmit} className="mb-8">
                    <div className="flex flex-col rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-8 transition-colors hover:border-blue-400">
                        <label
                            htmlFor="file-upload"
                            className="mb-3 cursor-pointer text-base font-semibold text-gray-700"
                        >
                            📁 Select Audio File
                        </label>
                        <input
                            id="file-upload"
                            type="file"
                            accept="audio/*,.wav,.mp3,.ogg,.flac"
                            onChange={handleFileChange}
                            className="mb-4 block w-full text-sm text-gray-500
                                             file:mr-4 file:rounded-lg file:border-0
                                             file:bg-blue-600 file:px-6
                                             file:py-3 file:text-sm
                                             file:font-semibold file:text-white
                                             hover:file:bg-blue-700 file:cursor-pointer"
                        />
                        {file && (
                            <p className="mb-3 text-sm text-gray-600">
                                Selected: <span className="font-medium">{file.name}</span>
                            </p>
                        )}
                        <button
                            type="submit"
                            disabled={!file || isLoading}
                            className="mt-2 w-full rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-3 text-lg font-semibold text-white shadow-md transition-all hover:from-blue-700 hover:to-indigo-700 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:from-blue-600 disabled:hover:to-indigo-600"
                        >
                            {isLoading ? '🔄 Analyzing...' : '🚀 Analyze Audio'}
                        </button>
                    </div>
                </form>

                {/* --- Loading Spinner --- */}
                {isLoading && (
                    <div className="flex flex-col items-center justify-center py-12">
                        <div
                            className="h-16 w-16 animate-spin rounded-full border-4 border-gray-200 border-t-blue-600"
                            role="status"
                        >
                            <span className="sr-only">Loading...</span>
                        </div>
                        <p className="mt-4 text-gray-600">Processing your audio file...</p>
                    </div>
                )}

                {/* --- Error Display --- */}
                {error && (
                    <div
                        className="rounded-lg border-l-4 border-red-500 bg-red-50 p-4 text-red-800"
                        role="alert"
                    >
                        <div className="flex">
                            <div className="flex-shrink-0">
                                <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <p className="text-sm font-medium">{error}</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* --- Results Display --- */}
                {result && (
                    <div className="space-y-6 animate-in fade-in duration-500">
                        {/* --- Audio Player --- */}
                        {audioURL && (
                            <div className="rounded-xl border border-gray-200 bg-gradient-to-r from-purple-50 to-pink-50 p-6 shadow-sm">
                                <h3 className="mb-3 text-xl font-bold text-gray-800">
                                    🎧 Audio Playback
                                </h3>
                                <audio controls src={audioURL} className="w-full" />
                            </div>
                        )}

                        {/* --- Top Predictions --- */}
                        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                            <h3 className="mb-4 text-xl font-bold text-gray-800">
                                🎯 Classification Results
                            </h3>
                            <div className="space-y-3">
                                {result.predictions.map((pred, idx) => (
                                    <div
                                        key={pred.index}
                                        className={`flex items-center justify-between rounded-lg p-4 transition-colors ${
                                            idx === 0
                                                ? 'bg-gradient-to-r from-blue-100 to-indigo-100 border-2 border-blue-300'
                                                : 'bg-gray-50 border border-gray-200'
                                        }`}
                                    >
                                        <div className="flex items-center space-x-3">
                                            <span className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold ${
                                                idx === 0 ? 'bg-blue-600 text-white' : 'bg-gray-300 text-gray-700'
                                            }`}>
                                                #{idx + 1}
                                            </span>
                                            <span className="font-semibold text-gray-800">
                                                {pred.label.replace(/_/g, ' ')}
                                            </span>
                                        </div>
                                        <div className="flex items-center space-x-3">
                                            <div className="h-2 w-32 overflow-hidden rounded-full bg-gray-200">
                                                <div
                                                    className={`h-full transition-all ${
                                                        idx === 0 ? 'bg-blue-600' : 'bg-gray-400'
                                                    }`}
                                                    style={{ width: `${pred.probability * 100}%` }}
                                                />
                                            </div>
                                            <span className="min-w-[4rem] text-right font-mono text-sm font-medium text-gray-700">
                                                {(pred.probability * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* --- Grad-CAM Visualization --- */}
                        {result.cam_base64 && (
                            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                                <h3 className="mb-3 text-xl font-bold text-gray-800">
                                    🧠 Grad-CAM Heatmap (Explainable AI)
                                </h3>
                                <p className="mb-4 text-sm text-gray-600">
                                    This visualization shows which parts of the audio spectrogram the AI model focused on when making its prediction. 
                                    Red areas indicate regions that strongly influenced the classification decision.
                                </p>
                                <div className="flex w-full justify-center overflow-hidden rounded-lg border border-gray-300 bg-gray-50">
                                    <img
                                        src={`data:image/png;base64,${result.cam_base64}`}
                                        alt="Grad-CAM Heatmap"
                                        className="max-w-full"
                                    />
                                </div>
                                <div className="mt-4 rounded-lg bg-blue-50 p-4">
                                    <p className="text-sm text-gray-700">
                                        <strong>💡 How to interpret:</strong> The heatmap overlays the mel spectrogram with warm colors (red/orange) 
                                        highlighting frequency-time regions that contributed most to predicting <strong className="text-blue-700">{result.predictions[0].label.replace(/_/g, ' ')}</strong>.
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* --- Analysis Info --- */}
                        <div className="rounded-xl border border-gray-200 bg-gradient-to-r from-green-50 to-teal-50 p-6 shadow-sm">
                            <h3 className="mb-3 text-xl font-bold text-gray-800">
                                ℹ️ Analysis Information
                            </h3>
                            <div className="grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
                                <div className="flex items-center space-x-2">
                                    <span className="font-semibold text-gray-700">Model:</span>
                                    <span className="text-gray-600">PANNs CNN14</span>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <span className="font-semibold text-gray-700">Classes:</span>
                                    <span className="text-gray-600">527 AudioSet categories</span>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <span className="font-semibold text-gray-700">XAI Method:</span>
                                    <span className="text-gray-600">Grad-CAM</span>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <span className="font-semibold text-gray-700">Processing Time:</span>
                                    <span className="text-gray-600">{result.prediction_time.toFixed(2)}s</span>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}