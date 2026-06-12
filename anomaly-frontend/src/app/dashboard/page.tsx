"use client";

import { useState, useEffect } from "react";
import { API_URL } from "@/lib/api";

interface DashboardStats {
  totalInferencesThisWeek: number;
  totalAnomaliesThisWeek: number;
  overallAnomalyRatePercentage: number;
  anomaliesByCategory: Record<string, number>;
  inferencesByDay: Record<string, number>;
}

const formatCategoryName = (cat: string) =>
  cat.split("_").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");

const formatDayLabel = (dateStr: string) => {
  const d = new Date(dateStr + "T00:00:00");
  return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
};

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStats>({
    totalInferencesThisWeek: 0,
    totalAnomaliesThisWeek: 0,
    overallAnomalyRatePercentage: 0,
    anomaliesByCategory: {},
    inferencesByDay: {},
  });

  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const fetchStatistics = async () => {
      try {
        const token = localStorage.getItem("token");
        const response = await fetch(`${API_URL}/api/v1/Statistics`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (response.ok) {
          const data = await response.json();
          setStats(data);
        } else {
          setErrorMessage("Failed to load dashboard statistics.");
        }
      } catch {
        setErrorMessage("Could not connect to the server.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatistics();
  }, []);

  // --- Derived chart data ---
  const sortedDays = Object.entries(stats.inferencesByDay).sort(([a], [b]) => a.localeCompare(b));
  const maxDayCount = Math.max(...sortedDays.map(([, v]) => v), 1);

  const sortedCategories = Object.entries(stats.anomaliesByCategory).sort(([, a], [, b]) => b - a);
  const maxCatCount = Math.max(...sortedCategories.map(([, v]) => v), 1);

  const Skeleton = ({ className }: { className: string }) => (
    <div className={`animate-pulse bg-gray-200 rounded ${className}`} />
  );

  return (
    <div className="space-y-6">
      {errorMessage && (
        <div className="p-4 bg-red-50 text-red-700 rounded-lg border border-red-200 text-sm">
          {errorMessage}
        </div>
      )}

      {/* ── Top Stat Cards ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Total Scans */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">Total scans (7 days)</p>
          {isLoading ? (
            <Skeleton className="h-10 w-24 mt-1" />
          ) : (
            <p className="text-4xl font-bold text-gray-800">{stats.totalInferencesThisWeek}</p>
          )}
        </div>

        {/* Total Anomalies */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-red-100">
          <p className="text-xs font-semibold uppercase tracking-widest text-red-400 mb-2">Anomalies detected</p>
          {isLoading ? (
            <Skeleton className="h-10 w-24 mt-1" />
          ) : (
            <p className="text-4xl font-bold text-red-600">{stats.totalAnomaliesThisWeek}</p>
          )}
        </div>

        {/* Defect Rate */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-orange-100">
          <p className="text-xs font-semibold uppercase tracking-widest text-orange-400 mb-2">Defect rate</p>
          {isLoading ? (
            <Skeleton className="h-10 w-24 mt-1" />
          ) : (
            <p className="text-4xl font-bold text-orange-500">{stats.overallAnomalyRatePercentage}%</p>
          )}
        </div>
      </div>

      {/* ── Bottom Row: Day chart + Category breakdown ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Inferences by Day */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-widest mb-5">
            Scans per day
          </h3>

          {isLoading ? (
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
                <Skeleton key={i} className="h-7 w-full" />
              ))}
            </div>
          ) : sortedDays.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40 text-gray-400 text-sm">
              <svg className="w-8 h-8 mb-2 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              No scan data for this period
            </div>
          ) : (
            <div className="space-y-2.5">
              {sortedDays.map(([day, count]) => (
                <div key={day} className="flex items-center gap-3">
                  <span className="text-xs text-gray-500 w-28 shrink-0">{formatDayLabel(day)}</span>
                  <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                    <div
                      className="h-6 rounded-full bg-blue-500 flex items-center justify-end pr-2 transition-all duration-500"
                      style={{ width: `${Math.max((count / maxDayCount) * 100, 8)}%` }}
                    >
                      <span className="text-xs font-semibold text-white">{count}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Anomalies by Category */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-widest mb-5">
            Anomalies by category
          </h3>

          {isLoading ? (
            <div className="space-y-3">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-7 w-full" />
              ))}
            </div>
          ) : sortedCategories.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-40 text-gray-400 text-sm">
              <svg className="w-8 h-8 mb-2 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              No anomalies detected this week
            </div>
          ) : (
            <div className="space-y-2.5">
              {sortedCategories.map(([category, count]) => (
                <div key={category} className="flex items-center gap-3">
                  <span
                    className="text-xs text-gray-500 w-28 shrink-0 truncate"
                    title={formatCategoryName(category)}
                  >
                    {formatCategoryName(category)}
                  </span>
                  <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                    <div
                      className="h-6 rounded-full bg-red-400 flex items-center justify-end pr-2 transition-all duration-500"
                      style={{ width: `${Math.max((count / maxCatCount) * 100, 8)}%` }}
                    >
                      <span className="text-xs font-semibold text-white">{count}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}