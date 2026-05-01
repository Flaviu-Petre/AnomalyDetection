"use client";

import { useState, useEffect } from "react";

interface ModelInfo {
  category: string;
  threshold: number;
}

export default function ModelsManagerPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [fetchError, setFetchError] = useState("");

  const [uploadCategory, setUploadCategory] = useState("");
  const [onnxFile, setOnnxFile] = useState<File | null>(null);
  const [onnxDataFile, setOnnxDataFile] = useState<File | null>(null);
  const [jsonFile, setJsonFile] = useState<File | null>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");

  const fetchModels = async () => {
    setIsLoading(true);
    setFetchError("");
    try {
      const token = localStorage.getItem("token");
      const response = await fetch("https://localhost:7136/api/v1/Models/get_all_models", {
        headers: { "Authorization": `Bearer ${token}` }
      });

      if (response.ok) {
        const data = await response.json();
        setModels(data);
      } else {
        setFetchError("Failed to load models. Are you logged in as an Admin?");
      }
    } catch (error) {
      setFetchError("Could not connect to the server.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadCategory || !onnxFile || !jsonFile || !onnxDataFile) {
      setUploadError("Category, .onnx file, .json file, and .onnx data file are strictly required.");
      return;
    }

    setIsUploading(true);
    setUploadError("");
    setUploadMessage("");

    try {
      const token = localStorage.getItem("token");
      const formData = new FormData();

      formData.append("Category", uploadCategory);
      formData.append("OnnxModel", onnxFile);
      formData.append("JsonMetadata", jsonFile);

      if (onnxDataFile) {
        formData.append("OnnxData", onnxDataFile);
      }

      const response = await fetch("https://localhost:7136/api/v1/Models/upload_model", {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}` },
        body: formData,
      });

      if (response.ok) {
        setUploadMessage(`Successfully uploaded AI model for '${uploadCategory}'!`);
        setUploadCategory("");
        setOnnxFile(null);
        setOnnxDataFile(null);
        setJsonFile(null);
        fetchModels();
      } else {
        const errText = await response.text();
        setUploadError(`Upload failed: ${errText}`);
      }
    } catch (error) {
      setUploadError("Could not connect to the server to upload.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (categoryToDelete: string) => {
    if (!window.confirm(`Are you absolutely sure you want to delete the model for '${categoryToDelete}'? This cannot be undone.`)) {
      return;
    }

    try {
      const token = localStorage.getItem("token");
      const response = await fetch(`https://localhost:7136/api/v1/Models/delete_category?category=${categoryToDelete}`, {
        method: "DELETE",
        headers: { "Authorization": `Bearer ${token}` }
      });

      if (response.ok) {
        fetchModels();
      } else {
        alert("Failed to delete model. Check server logs.");
      }
    } catch (error) {
      alert("Network error while trying to delete.");
    }
  };

  const formatCategoryName = (cat: string) => {
    return cat.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  return (
    <div className="max-w-400 mx-auto space-y-8">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800">AI model management</h2>
        <p className="text-gray-600 mt-1">Upload new neural networks or manage existing ones.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* LEFT COLUMN: Upload Form */}
        <div className="lg:col-span-1 bg-white p-6 rounded-xl shadow-lg border border-gray-200 h-fit">
          <h3 className="text-lg font-semibold text-gray-800 mb-6 flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
            Upload new model
          </h3>

          <form onSubmit={handleUpload} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Category name</label>
              <input
                type="text"
                placeholder="e.g., bottle, metal_nut, carpet"
                value={uploadCategory}
                onChange={(e) => setUploadCategory(e.target.value.toLowerCase().replace(/\s+/g, '_'))}
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-gray-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-sm"
                required
              />
              <p className="text-xs text-gray-500 mt-1">Use lowercase and underscores.</p>
            </div>

            <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">1. ONNX model file <span className="text-red-500">*</span></label>
                <input
                  type="file"
                  accept=".onnx"
                  onChange={(e) => setOnnxFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">2. JSON metadata <span className="text-red-500">*</span></label>
                <input
                  type="file"
                  accept=".json"
                  onChange={(e) => setJsonFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">3. ONNX data file <span className="text-red-500">*</span></label>
                <input
                  type="file"
                  onChange={(e) => setOnnxDataFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isUploading || !uploadCategory || !onnxFile || !jsonFile || !onnxDataFile}
              className="w-full py-2.5 bg-blue-700 text-white font-semibold rounded-md shadow hover:bg-blue-800 disabled:bg-gray-400 transition-colors flex justify-center items-center gap-2"
            >
              {isUploading ? (
                <><svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Uploading to server...</>
              ) : "Upload model"}
            </button>

            {uploadError && <div className="p-3 bg-red-50 text-red-700 rounded text-sm border border-red-200">{uploadError}</div>}
            {uploadMessage && <div className="p-3 bg-green-50 text-green-700 rounded text-sm border border-green-200">{uploadMessage}</div>}
          </form>
        </div>

        {/* RIGHT COLUMN: Active models list */}
        <div className="lg:col-span-2 bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden flex flex-col h-full">
          <div className="p-6 border-b border-gray-200 bg-gray-50 flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
              Active models on server
            </h3>
            <button onClick={fetchModels} className="text-sm text-blue-600 hover:text-blue-800 font-medium">Refresh list</button>
          </div>

          <div className="p-0 flex-1 overflow-x-auto">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center p-12 text-gray-400">
                <svg className="animate-spin h-8 w-8 mb-4 text-blue-500" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                <p>Loading models...</p>
              </div>
            ) : fetchError ? (
              <div className="p-6 text-center text-red-500">{fetchError}</div>
            ) : models.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-16 text-gray-400 text-center">
                <svg className="w-12 h-12 mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path></svg>
                <p className="text-lg font-medium text-gray-600">No models found</p>
                <p className="text-sm mt-1">Upload a model using the form to get started.</p>
              </div>
            ) : (
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-gray-50 border-b border-gray-200 text-xs uppercase tracking-wider text-gray-500">
                    <th className="p-4 font-semibold">Category</th>
                    <th className="p-4 font-semibold">Configured threshold</th>
                    <th className="p-4 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {models.map((m) => (
                    <tr key={m.category} className="hover:bg-gray-50 transition-colors">
                      <td className="p-4">
                        <span className="font-semibold text-gray-800">{formatCategoryName(m.category)}</span>
                        <span className="block text-xs text-gray-500 mt-0.5 font-mono">{m.category}</span>
                      </td>
                      <td className="p-4">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 border border-blue-200">
                          {m.threshold.toFixed(4)}
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        <button
                          onClick={() => handleDelete(m.category)}
                          className="text-red-500 hover:text-red-700 hover:bg-red-50 p-2 rounded-md transition-colors inline-flex items-center gap-1 text-sm font-medium"
                          title="Delete model"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}