"use client";

import { useRef, useState } from "react";

export default function InferencePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [requestHeatmap, setRequestHeatmap] = useState(true);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- EVENT HANDLERS ---
  const runInference = async (file: File, heatmap: boolean) => {
    setIsLoading(true);
    setErrorMessage("");
    setResult(null);
    setFeedbackSubmitted(false);
    setFeedbackMessage("");

    try {
      const token = localStorage.getItem("token");

      const formData = new FormData();
      formData.append("image", file);
      formData.append("returnHeatmap", heatmap.toString());

      const response = await fetch("https://localhost:7136/api/v1/Inference/detect_anomaly", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setResult(data);
      } else if (response.status === 401) {
        setErrorMessage("Unauthorized. Your session may have expired.");
      } else {
        const errText = await response.text();
        setErrorMessage(`Error: ${errText || "Failed to analyze the image."}`);
      }
    } catch (error) {
      setErrorMessage("Could not connect to the server. Is the API running?");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setErrorMessage("");
      setFeedbackSubmitted(false);
      setFeedbackMessage("");
      runInference(file, requestHeatmap);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;
    runInference(selectedFile, requestHeatmap);
  };

  const handleFeedback = async (isCorrect: boolean) => {
    if (!selectedFile || !result || !result.predictedCategory) return;

    setIsSubmittingFeedback(true);
    setFeedbackMessage("");

    try {
      const token = localStorage.getItem("token");

      const actualAnomalyState = isCorrect ? result.isAnomaly : !result.isAnomaly;

      const formData = new FormData();
      formData.append("category", result.predictedCategory);
      formData.append("image", selectedFile);
      formData.append("isActuallyAnomaly", actualAnomalyState.toString());

      const response = await fetch("https://localhost:7136/api/v1/Feedback", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`
        },
        body: formData
      });

      if (response.ok) {
        setFeedbackSubmitted(true);
        setFeedbackMessage("Thank you! Feedback saved.");
      } else {
        setFeedbackMessage("Failed to submit feedback. Please try again.");
      }
    } catch (error) {
      setFeedbackMessage("Could not connect to server to submit feedback.");
    } finally {
      setIsSubmittingFeedback(false);
    }
  };

  const formatCategoryName = (cat: string) => {
    if (!cat) return "Unknown";
    return cat.split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // --- UI RENDER ---
  return (
    <div className="max-w-400 mx-auto space-y-8">

      {/* Upload and configuration bar */}
      <div className="bg-white px-6 py-4 rounded-xl shadow-lg border border-gray-200">
        <div className="flex flex-wrap items-center gap-4">

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept="image/*"
            onChange={handleFileChange}
          />

          {/* Upload button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 border-2 border-dashed border-gray-300 text-gray-600 text-sm font-medium rounded-lg hover:border-blue-400 hover:text-blue-600 hover:bg-blue-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Upload image
          </button>

          {selectedFile ? (
            <span className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 border border-blue-200 text-blue-700 text-sm rounded-md font-medium max-w-xs truncate">
              <svg className="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
              </svg>
              <span className="truncate">{selectedFile.name}</span>
            </span>
          ) : (
            <span className="text-sm text-gray-400">No image selected — PNG, JPG or JPEG</span>
          )}

          <div className="flex-1" />

          <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
            <input
              type="checkbox"
              id="heatmap-toggle"
              checked={requestHeatmap}
              onChange={(e) => setRequestHeatmap(e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
            />
            Generate heatmap
          </label>

          <button
            onClick={handleRunInference}
            disabled={!selectedFile || isLoading}
            className="flex items-center gap-2 px-5 py-2 bg-blue-700 text-white text-sm font-semibold rounded-lg shadow hover:bg-blue-800 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Analyzing...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Run AI inference
              </>
            )}
          </button>
        </div>

        {errorMessage && (
          <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md border border-red-200 text-sm font-medium">
            {errorMessage}
          </div>
        )}
      </div>

      {/* Results section */}
      {(previewUrl || result || isLoading) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

          {/* COLUMN 1: Source Image */}
          <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
            <h3 className="text-sm font-semibold text-gray-500 mb-4 uppercase tracking-wider">Source Image</h3>
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" className="w-full h-auto rounded-lg border border-gray-200 shadow-sm" />
            ) : (
              <div className="w-full aspect-square bg-gray-50 border border-gray-200 rounded-lg flex items-center justify-center text-gray-400 text-sm">
                Awaiting upload
              </div>
            )}
          </div>

          {/* COLUMN 2: Analysis result */}
          <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200 flex flex-col h-full">
            <h3 className="text-sm font-semibold text-gray-500 mb-4 uppercase tracking-wider">Analysis result</h3>

            {isLoading && (
              <div className="flex-1 flex flex-col items-center justify-center text-blue-600 animate-pulse py-12">
                <svg className="w-16 h-16 mb-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <p className="font-semibold text-lg">AI is processing...</p>
              </div>
            )}

            {!result && !isLoading && (
              <div className="flex-1 flex items-center justify-center text-gray-400 py-12 text-sm text-center">
                Upload an image to start analysis.
              </div>
            )}

            {result && !isLoading && (
              <div className="flex flex-col justify-center items-center text-center space-y-5 flex-1">

                <div className="w-full mb-2">
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">AI detected category</p>
                  <span className="inline-block px-4 py-1.5 bg-blue-100 text-blue-800 font-bold rounded-md border border-blue-200 text-sm">
                    {formatCategoryName(result.predictedCategory)}
                  </span>
                </div>

                <div className={`text-3xl font-bold px-8 py-4 rounded-full w-full border ${result.isAnomaly ? 'bg-red-100 text-red-700 border-red-200' : 'bg-green-100 text-green-700 border-green-200'}`}>
                  {result.isAnomaly ? "ANOMALY" : "NORMAL"}
                </div>

                <div className="w-full text-left space-y-4 bg-gray-50 p-6 rounded-lg border border-gray-200 flex-1">
                  <p className="text-base text-gray-600 flex justify-between border-b pb-3">
                    <strong>Anomaly score:</strong>
                    <span className="text-gray-900 font-mono text-lg">{result.score?.toFixed(4) || "N/A"}</span>
                  </p>
                  <p className="text-base text-gray-600 flex justify-between">
                    <strong>Threshold limit:</strong>
                    <span className="text-gray-900 font-mono text-lg">{result.usedThreshold?.toFixed(4) || "N/A"}</span>
                  </p>
                </div>

                {/* Feedback */}
                <div className="w-full mt-4 pt-4 border-t border-gray-100">
                  {!feedbackSubmitted ? (
                    <>
                      <p className="text-sm font-medium text-gray-600 mb-3">Did the AI get this right?</p>
                      <div className="flex justify-center gap-3">
                        <button
                          onClick={() => handleFeedback(true)}
                          disabled={isSubmittingFeedback}
                          className="px-5 py-2 bg-gray-100 hover:bg-green-100 hover:text-green-800 text-gray-700 text-sm font-semibold rounded-md transition-colors disabled:opacity-50"
                        >
                          Yes
                        </button>
                        <button
                          onClick={() => handleFeedback(false)}
                          disabled={isSubmittingFeedback}
                          className="px-5 py-2 bg-gray-100 hover:bg-red-100 hover:text-red-800 text-gray-700 text-sm font-semibold rounded-md transition-colors disabled:opacity-50"
                        >
                          No (Report)
                        </button>
                      </div>
                      {feedbackMessage && <p className="text-red-600 text-xs mt-3 font-medium">{feedbackMessage}</p>}
                    </>
                  ) : (
                    <div className="flex items-center justify-center text-green-700 text-sm font-medium gap-2 p-3 bg-green-50 rounded-md border border-green-200">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                      {feedbackMessage}
                    </div>
                  )}
                </div>

              </div>
            )}
          </div>

          {/* COLUMN 3: Heatmap */}
          <div className={`p-6 rounded-xl shadow-lg border ${result?.isAnomaly && result?.heatmapBase64 ? 'border-red-300 bg-red-50' : 'bg-white border-gray-200'}`}>
            <h3 className={`text-sm font-semibold mb-4 uppercase tracking-wider flex items-center gap-2 ${result?.isAnomaly && result?.heatmapBase64 ? 'text-red-700' : 'text-gray-500'}`}>
              AI Heatmap
            </h3>

            {result?.heatmapBase64 ? (
              <img
                src={`data:image/jpeg;base64,${result.heatmapBase64}`}
                alt="AI Detection Heatmap"
                className="w-full h-auto rounded-lg border border-red-200 shadow-md"
              />
            ) : isLoading ? (
              <div className="w-full aspect-square bg-gray-50 border border-gray-200 rounded-lg flex flex-col items-center justify-center text-gray-400 text-sm animate-pulse text-center p-4">
                <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Generating heatmap...
              </div>
            ) : !result ? (
              <div className="w-full aspect-square bg-gray-50 border border-gray-200 rounded-lg flex items-center justify-center text-gray-400 text-sm">
                Awaiting analysis
              </div>
            ) : !requestHeatmap ? (
              <div className="w-full aspect-square bg-yellow-50 border border-yellow-200 rounded-lg flex flex-col items-center justify-center text-yellow-700 text-sm text-center p-4">
                <svg className="w-12 h-12 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                Heatmap disabled in configuration.
              </div>
            ) : (
              <div className="w-full aspect-square bg-gray-50 border border-gray-200 rounded-lg flex items-center justify-center text-gray-400 text-sm">
                Not available
              </div>
            )}
          </div>

        </div>
      )}
    </div>
  );
}