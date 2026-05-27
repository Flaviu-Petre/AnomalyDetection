"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface CategorySummary {
  category: string;
  anomalyCount: number;
  goodCount: number;
}

export default function FeedbackPage() {
  const router = useRouter();
  const [summary, setSummary] = useState<CategorySummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<"anomaly" | "good" | null>(null);
  const [imageNames, setImageNames] = useState<string[]>([]);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [blobUrls, setBlobUrls] = useState<Record<string, string>>({});

  const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const role = typeof window !== "undefined" ? localStorage.getItem("role") : null;

  useEffect(() => {
    if (role !== "Admin") {
      router.push("/dashboard");
      return;
    }
    fetchSummary();
  }, []);

  const fetchSummary = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("https://localhost:7136/api/v1/Feedback/summary", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setSummary(data);
      } else {
        setErrorMessage("Failed to load feedback summary.");
      }
    } catch {
      setErrorMessage("Could not connect to the server.");
    } finally {
      setIsLoading(false);
    }
  };

  const loadImage = async (category: string, label: string, filename: string) => {
    const key = `${category}/${label}/${filename}`;
    if (blobUrls[key]) return;

    try {
      const response = await fetch(
        `https://localhost:7136/api/v1/Feedback/images/${category}/${label}/${filename}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setBlobUrls(prev => ({ ...prev, [key]: url }));
      }
    } catch {
        setErrorMessage("Failed to load image. Please try again.");
    }
  };

  const openGallery = async (category: string, label: "anomaly" | "good") => {
    setSelectedCategory(category);
    setSelectedLabel(label);
    setImageNames([]);
    setBlobUrls({});
    setIsLoadingImages(true);

    try {
      const response = await fetch(
        `https://localhost:7136/api/v1/Feedback/images/${category}/${label}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (response.ok) {
        const data: string[] = await response.json();
        setImageNames(data);
        for (const name of data) {
          loadImage(category, label, name);
        }
      }
    } catch {
      setErrorMessage("Failed to load images. Please try again.");
    } finally {
      setIsLoadingImages(false);
    }
  };

  const closeGallery = () => {
    Object.values(blobUrls).forEach(url => URL.revokeObjectURL(url));
    setSelectedCategory(null);
    setSelectedLabel(null);
    setImageNames([]);
    setBlobUrls({});
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-xl font-semibold text-gray-800">Feedback data</h3>
        <p className="text-sm text-gray-500 mt-1">
          Images submitted by operators for feedback.
        </p>
      </div>

      {errorMessage && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {errorMessage}
        </div>
      )}

      {/* SUMMARY TABLE */}
      {isLoading ? (
        <div className="flex items-center justify-center p-16 text-gray-400">
          <svg className="animate-spin h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading...
        </div>
      ) : summary.length === 0 ? (
        <div className="p-12 text-center text-gray-400 bg-white rounded-xl border border-gray-200">
          No feedback data collected yet.
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">Category</th>
                <th className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider text-center">Anomaly images</th>
                <th className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider text-center">Good images</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {summary.map((row) => (
                <tr key={row.category} className="hover:bg-gray-50 transition-colors">
                  <td className="p-4 text-sm font-semibold text-gray-900 capitalize">{row.category}</td>
                  <td className="p-4 text-center">
                    <button
                      onClick={() => openGallery(row.category, "anomaly")}
                      disabled={row.anomalyCount === 0}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200 hover:bg-red-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {row.anomalyCount} {row.anomalyCount === 1 ? "image" : "images"}
                    </button>
                  </td>
                  <td className="p-4 text-center">
                    <button
                      onClick={() => openGallery(row.category, "good")}
                      disabled={row.goodCount === 0}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-200 hover:bg-green-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {row.goodCount} {row.goodCount === 1 ? "image" : "images"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* GALLERY MODAL */}
      {selectedCategory && selectedLabel && (
        <div
          className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6"
          onClick={closeGallery}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <div>
                <h4 className="text-lg font-semibold text-gray-900 capitalize">
                  {selectedCategory} —{" "}
                  <span className={selectedLabel === "anomaly" ? "text-red-600" : "text-green-600"}>
                    {selectedLabel === "anomaly" ? "Anomaly" : "Good"}
                  </span>
                </h4>
                <p className="text-sm text-gray-500 mt-0.5">{imageNames.length} {imageNames.length === 1 ? "image" : "images"}</p>
              </div>
              <button
                onClick={closeGallery}
                className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Image grid */}
            <div className="overflow-y-auto p-6">
              {isLoadingImages ? (
                <div className="flex items-center justify-center p-12 text-gray-400">
                  <svg className="animate-spin h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Loading images...
                </div>
              ) : imageNames.length === 0 ? (
                <p className="text-center text-gray-400 py-12">No images found.</p>
              ) : (
                <div className="grid grid-cols-3 gap-4">
                  {imageNames.map((name) => {
                    const key = `${selectedCategory}/${selectedLabel}/${name}`;
                    return (
                      <div key={name} className="rounded-lg overflow-hidden border border-gray-200 bg-gray-50">
                        {blobUrls[key] ? (
                          <img
                            src={blobUrls[key]}
                            alt={name}
                            className="w-full h-48 object-contain"
                          />
                        ) : (
                          <div className="w-full h-48 flex items-center justify-center text-gray-300">
                            <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                          </div>
                        )}
                        <p className="text-xs text-gray-400 px-2 py-1 truncate">{name}</p>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}