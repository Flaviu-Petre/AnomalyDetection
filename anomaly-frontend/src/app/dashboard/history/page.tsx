"use client";

import { useState, useEffect } from "react";
import { API_URL } from "@/lib/api";

interface InferenceRecord {
  id: number;
  timestamp: string;
  category: string;
  isAnomaly: boolean;
  score: number;
  thresholdUsed: number;
  username: string;
  imageName: string;
}

const CATEGORIES = [
  "bottle", "cable", "capsule", "carpet", "grid",
  "hazelnut", "leather", "metal_nut", "pill", "screw",
  "tile", "toothbrush", "transistor", "wood", "zipper"
];

export default function HistoryPage() {
  const [records, setRecords] = useState<InferenceRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");
  const [userRole, setUserRole] = useState<string>("");

  // Pagination
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [totalPages, setTotalPages] = useState(1);
  const [totalCount, setTotalCount] = useState(0);

  // Sorting
  const [sortBy, setSortBy] = useState("timestamp");
  const [sortDesc, setSortDesc] = useState(true);

  // Filters
  const [filterAnomaly, setFilterAnomaly] = useState<string>("");
  const [filterCategory, setFilterCategory] = useState<string>("");
  const [filterUsername, setFilterUsername] = useState<string>("");

  useEffect(() => {
    const role = localStorage.getItem("role") ?? "";
    setUserRole(role);
  }, []);

  useEffect(() => {
    const fetchHistory = async () => {
      setIsLoading(true);
      setErrorMessage("");
      try {
        const token = localStorage.getItem("token");

        const params = new URLSearchParams({
          page: String(page),
          pageSize: String(pageSize),
          sortBy,
          sortDesc: String(sortDesc),
        });

        if (filterAnomaly !== "") params.append("isAnomaly", filterAnomaly);
        if (filterCategory !== "") params.append("category", filterCategory);
        if (filterUsername.trim() !== "" && userRole === "Admin")
          params.append("filterUsername", filterUsername.trim());

        const url = `${API_URL}/api/v1/Statistics/history?${params.toString()}`;

        const response = await fetch(url, {
          headers: { "Authorization": `Bearer ${token}` }
        });

        if (response.ok) {
          const data = await response.json();
          setRecords(data.items);
          setTotalPages(data.totalPages);
          setTotalCount(data.totalCount);
        } else {
          setErrorMessage("Failed to load inference history.");
        }
      } catch {
        setErrorMessage("Could not connect to the server.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, [page, pageSize, sortBy, sortDesc, filterAnomaly, filterCategory, filterUsername, userRole]);

  const formatCategory = (cat: string) =>
    cat.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");

  const formatDate = (dateString: string) =>
    new Date(dateString).toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
    });

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortDesc(!sortDesc);
    } else {
      setSortBy(column);
      setSortDesc(true);
    }
    setPage(1);
  };

  const handleFilterChange = (setter: (v: string) => void, value: string) => {
    setter(value);
    setPage(1);
  };

  const clearFilters = () => {
    setFilterAnomaly("");
    setFilterCategory("");
    setFilterUsername("");
    setPage(1);
  };

  const hasActiveFilters =
    filterAnomaly !== "" || filterCategory !== "" || filterUsername !== "";

  const SortArrow = ({ column }: { column: string }) => {
    if (sortBy !== column)
      return <span className="opacity-0 group-hover:opacity-30 ml-1">↕</span>;
    return <span className="ml-1 text-blue-500">{sortDesc ? "↓" : "↑"}</span>;
  };

  return (
    <div className="max-w-400 mx-auto space-y-8">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800">Inference history</h2>
        <p className="text-gray-600 mt-1">A detailed audit log of all recent AI inspections.</p>
      </div>

      {errorMessage && (
        <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-md border border-red-200 text-sm">
          {errorMessage}
        </div>
      )}

      <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden flex flex-col">

        {/* FILTER BAR */}
        <div className="flex flex-wrap items-center gap-3 px-6 py-4 border-b border-gray-200 bg-gray-50">

          {/* Results */}
          <select
            value={filterAnomaly}
            onChange={e => handleFilterChange(setFilterAnomaly, e.target.value)}
            className="text-sm border border-gray-300 rounded-md px-3 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            <option value="">All results</option>
            <option value="false">Normal only</option>
            <option value="true">Anomaly only</option>
          </select>

          {/* Category */}
          <select
            value={filterCategory}
            onChange={e => handleFilterChange(setFilterCategory, e.target.value)}
            className="text-sm border border-gray-300 rounded-md px-3 py-1.5 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
          >
            <option value="">All categories</option>
            {CATEGORIES.map(cat => (
              <option key={cat} value={cat}>{formatCategory(cat)}</option>
            ))}
          </select>

          {/* Filter by username (Admin only) */}
          {userRole === "Admin" && (
            <input
              type="text"
              placeholder="Filter by username"
              value={filterUsername}
              onChange={e => handleFilterChange(setFilterUsername, e.target.value)}
              className="text-sm border border-gray-300 rounded-md px-3 py-1.5 w-44 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          )}

          {/* Clear filters button */}
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="text-sm text-red-500 hover:text-red-700 font-medium px-2 transition-colors"
            >
              Clear filters
            </button>
          )}

          {/* Active results counter */}
          {!isLoading && hasActiveFilters && (
            <span className="ml-auto text-xs text-gray-500 font-medium">
              {totalCount} result{totalCount !== 1 ? "s" : ""} found
            </span>
          )}
        </div>

        {/* TABLE */}
        <div className="overflow-x-auto">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400">
              <svg className="animate-spin h-8 w-8 mb-4 text-blue-500" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <p>Loading audit trail...</p>
            </div>
          ) : records.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400 text-center">
              <svg className="w-12 h-12 mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
              </svg>
              <p className="text-lg font-medium text-gray-600">No history found</p>
              <p className="text-sm mt-1">
                {hasActiveFilters
                  ? "No records match the active filters."
                  : "Run new inferences to see data here."}
              </p>
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-xs uppercase tracking-wider text-gray-500 select-none">
                  <th className="p-4 font-semibold cursor-pointer group hover:bg-gray-100 transition-colors" onClick={() => handleSort("timestamp")}>
                    Date & Time <SortArrow column="timestamp" />
                  </th>
                  <th className="p-4 font-semibold cursor-pointer group hover:bg-gray-100 transition-colors" onClick={() => handleSort("operator")}>
                    Operator <SortArrow column="operator" />
                  </th>
                  <th className="p-4 font-semibold cursor-pointer group hover:bg-gray-100 transition-colors" onClick={() => handleSort("category")}>
                    Part category <SortArrow column="category" />
                  </th>
                  <th className="p-4 font-semibold">
                    Image name
                  </th>
                  <th className="p-4 font-semibold cursor-pointer group hover:bg-gray-100 transition-colors" onClick={() => handleSort("isanomaly")}>
                    AI decision <SortArrow column="isanomaly" />
                  </th>
                  <th className="p-4 font-semibold text-right cursor-pointer group hover:bg-gray-100 transition-colors" onClick={() => handleSort("score")}>
                    Confidence score <SortArrow column="score" />
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 text-sm">
                {records.map((record) => (
                  <tr key={record.id} className="hover:bg-gray-50 transition-colors">
                    <td className="p-4 text-gray-600 whitespace-nowrap">
                      {formatDate(record.timestamp)}
                    </td>
                    <td className="p-4">
                      <span className="font-medium text-gray-800 bg-gray-100 px-2 py-1 rounded-md">
                        @{record.username}
                      </span>
                    </td>
                    <td className="p-4 font-medium text-gray-800">
                      {formatCategory(record.category)}
                    </td>
                    <td className="p-4 text-gray-500 text-sm">
                      {record.imageName ? (
                        <span className="flex items-center gap-1 bg-gray-50 px-2 py-1 rounded border border-gray-100 w-fit">
                          <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                          <span className="truncate max-w-37.5" title={record.imageName}>
                            {record.imageName}
                          </span>
                        </span>
                      ) : (
                        <span className="text-gray-300 italic">Unknown</span>
                      )}
                    </td>
                    <td className="p-4">
                      {record.isAnomaly ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">
                          <span className="w-1.5 h-1.5 rounded-full bg-red-600" /> Anomaly
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-200">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-600" /> Normal
                        </span>
                      )}
                    </td>
                    <td className="p-4 text-right">
                      <div className="flex flex-col items-end">
                        <span className="font-mono font-medium text-gray-900">{record.score.toFixed(4)}</span>
                        <span className="text-xs text-gray-500 font-mono">Limit: {record.thresholdUsed.toFixed(4)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* PAGINATION FOOTER */}
        {!isLoading && records.length > 0 && (
          <div className="border-t border-gray-200 bg-gray-50 px-6 py-4 flex items-center justify-between">
            <div className="text-sm text-gray-500">
              Showing{" "}
              <span className="font-medium text-gray-900">{((page - 1) * pageSize) + 1}</span>
              {" "}to{" "}
              <span className="font-medium text-gray-900">{Math.min(page * pageSize, totalCount)}</span>
              {" "}of{" "}
              <span className="font-medium text-gray-900">{totalCount}</span> results
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-3 py-1.5 rounded border border-gray-300 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Previous
              </button>
              <div className="text-sm text-gray-600 px-2 font-medium">
                Page {page} of {totalPages}
              </div>
              <button
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
                className="px-3 py-1.5 rounded border border-gray-300 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}