"use client";

import { useState } from "react";

export default function InferencePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  const [category, setCategory] = useState("bottle");
  const [requestHeatmap, setRequestHeatmap] = useState(true);
  
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setErrorMessage("");
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setErrorMessage("");
    setResult(null);

    try {
      const token = localStorage.getItem("token");

      const formData = new FormData();
      formData.append("category", category);
      formData.append("image", selectedFile); 
      formData.append("returnHeatmap", requestHeatmap.toString());

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

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      
      {/* Upload & Configuration Card */}
      <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-6">Run AI Inspection</h3>
        
        {/* Configuration Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Part Category</label>
            <select 
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 bg-white shadow-sm"
            >
              <option value="bottle">Bottle</option>
              <option value="cable">Cable</option>
              <option value="capsule">Capsule</option>
              <option value="hazelnut">Hazelnut</option>
              <option value="metal_nut">Metal Nut</option>
              <option value="pill">Pill</option>
            </select>
          </div>
          
          <div className="flex items-center mt-6">
            <input 
              type="checkbox" 
              id="heatmap-toggle"
              checked={requestHeatmap}
              onChange={(e) => setRequestHeatmap(e.target.checked)}
              className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
            />
            <label htmlFor="heatmap-toggle" className="ml-3 block text-sm font-medium text-gray-700 cursor-pointer">
              Generate visual heatmap
              <span className="block text-xs text-gray-500 font-normal mt-0.5">Uncheck for faster performance</span>
            </label>
          </div>
        </div>

        {/* Image Dropzone */}
        <div className="flex flex-col items-center justify-center w-full">
          <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
            <div className="flex flex-col items-center justify-center pt-5 pb-6 text-gray-500">
              <svg className="w-10 h-10 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
              <p className="mb-2 text-sm"><span className="font-semibold">Click to upload</span> or drag and drop</p>
              <p className="text-xs">PNG, JPG or JPEG</p>
            </div>
            <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
          </label>
        </div>

        {/* Action Button */}
        <div className="mt-6 flex justify-end">
          <button
            onClick={handleRunInference}
            disabled={!selectedFile || isLoading}
            className="px-6 py-2 bg-blue-700 text-white font-semibold rounded-md shadow-sm hover:bg-blue-800 disabled:bg-gray-400 transition-colors"
          >
            {isLoading ? "Analyzing..." : "Run AI Inference"}
          </button>
        </div>

        {errorMessage && (
          <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-md border border-red-200 text-sm font-medium">
            {errorMessage}
          </div>
        )}
      </div>

      {/* Results Section */}
      {(previewUrl || result) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
          
          {/* LEFT COLUMN: Images */}
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
              <h3 className="text-sm font-semibold text-gray-500 mb-4 uppercase tracking-wider">Original image</h3>
              {previewUrl && (
                <img src={previewUrl} alt="Preview" className="w-full h-auto rounded-lg border border-gray-200" />
              )}
            </div>

            {/* Heatmap */}
            {result?.heatmapBase64 && (
              <div className="p-6 rounded-xl shadow-sm border border-red-200 bg-red-50">
                <h3 className="text-sm font-semibold text-red-600 mb-4 uppercase tracking-wider flex items-center gap-2">
                  <span>🔥</span> Anomaly Heatmap
                </h3>
                <img 
                  src={`data:image/jpeg;base64,${result.heatmapBase64}`} 
                  alt="AI Detection Heatmap" 
                  className="w-full h-auto rounded-lg border border-red-200 shadow-sm" 
                />
              </div>
            )}
          </div>

          {/* AI Result Readout */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 flex flex-col h-fit">
            <h3 className="text-sm font-semibold text-gray-500 mb-4 uppercase tracking-wider">Analysis result</h3>
            
            {!result && isLoading && (
              <div className="flex-1 flex flex-col items-center justify-center text-blue-600 animate-pulse py-12">
                <svg className="w-12 h-12 mb-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                <p className="font-semibold">Processing image through neural network...</p>
              </div>
            )}

            {result && (
              <div className="flex flex-col justify-center items-center text-center space-y-4 mt-2">
                <div className={`text-2xl font-bold px-6 py-3 rounded-full w-full ${result.isAnomaly ? 'bg-red-100 text-red-700 border border-red-200' : 'bg-green-100 text-green-700 border border-green-200'}`}>
                  {result.isAnomaly ? "ANOMALY DETECTED" : "NORMAL"}
                </div>
                
                <div className="w-full text-left space-y-3 mt-4 bg-gray-50 p-5 rounded-lg border border-gray-200">
                  <p className="text-sm text-gray-600 flex justify-between border-b pb-2">
                    <strong>Anomaly Score:</strong> 
                    <span className="text-gray-900 font-mono">{result.score?.toFixed(4) || "N/A"}</span>
                  </p>
                  <p className="text-sm text-gray-600 flex justify-between border-b pb-2">
                    <strong>Threshold Limit:</strong> 
                    <span className="text-gray-900 font-mono">{result.usedThreshold?.toFixed(4) || "N/A"}</span>
                  </p>
                </div>
              </div>
            )}
          </div>

        </div>
      )}

    </div>
  );
}