"use client";

import { useState, useEffect } from "react";

export default function DashboardPage() {
  const [stats, setStats] = useState({
    totalInferencesThisWeek: 0,
    totalAnomaliesThisWeek: 0,
    overallAnomalyRatePercentage: 0,
  });
  
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const fetchStatistics = async () => {
      try {
        const token = localStorage.getItem("token");
        
        const response = await fetch("https://localhost:7136/api/v1/Statistics", {
          headers: {
            "Authorization": `Bearer ${token}`
          }
        });

        if (response.ok) {
          const data = await response.json();
          setStats(data);
        } else {
          setErrorMessage("Failed to load dashboard statistics.");
        }
      } catch (error) {
        setErrorMessage("Could not connect to the server.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatistics();
  }, []);

  return (
    <>
      {errorMessage && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-md border border-red-200 text-sm">
          {errorMessage}
        </div>
      )}

      {/* Top Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        
        {/* Total Scans */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 flex flex-col justify-center">
          <h3 className="text-gray-500 text-sm font-medium uppercase tracking-wider">Total Scans (7 Days)</h3>
          <p className="text-4xl font-bold text-gray-800 mt-3">
            {isLoading ? (
              <span className="animate-pulse text-gray-300">...</span>
            ) : (
              stats.totalInferencesThisWeek
            )}
          </p>
        </div>

        {/* Total Anomalies */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-red-100 flex flex-col justify-center">
          <h3 className="text-red-500 text-sm font-medium uppercase tracking-wider">Anomalies Detected</h3>
          <p className="text-4xl font-bold text-red-600 mt-3">
            {isLoading ? (
              <span className="animate-pulse text-red-300">...</span>
            ) : (
              stats.totalAnomaliesThisWeek
            )}
          </p>
        </div>

        {/* Defect Rate */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-orange-100 flex flex-col justify-center">
          <h3 className="text-orange-500 text-sm font-medium uppercase tracking-wider">Defect Rate</h3>
          <p className="text-4xl font-bold text-orange-500 mt-3">
            {isLoading ? (
              <span className="animate-pulse text-orange-300">...</span>
            ) : (
              `${stats.overallAnomalyRatePercentage}%`
            )}
          </p>
        </div>
      </div>

      {/* Information Banner */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
        <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
        </div>
        <h3 className="text-lg font-semibold text-gray-800">System monitoring active</h3>
        <p className="text-gray-500 mt-2 max-w-md mx-auto">
          These statistics reflect the last 7 days of inference data. 
          Upload more images in the Run Inference tab to populate this dashboard.
        </p>
      </div>
    </>
  );
}