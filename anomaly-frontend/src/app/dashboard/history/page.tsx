"use client";

import { useState, useEffect } from "react";

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

export default function HistoryPage() {
  const [records, setRecords] = useState<InferenceRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const token = localStorage.getItem("token");
        const response = await fetch("https://localhost:7136/api/v1/Statistics/history", {
          headers: {
            "Authorization": `Bearer ${token}`
          }
        });

        if (response.ok) {
          const data = await response.json();
          setRecords(data);
        } else {
          setErrorMessage("Failed to load inference history.");
        }
      } catch (error) {
        setErrorMessage("Could not connect to the server.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const formatCategory = (cat: string) => cat.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
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
        <div className="overflow-x-auto">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400">
              <svg className="animate-spin h-8 w-8 mb-4 text-blue-500" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
              <p>Loading audit trail...</p>
            </div>
          ) : records.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400 text-center">
              <svg className="w-12 h-12 mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path></svg>
              <p className="text-lg font-medium text-gray-600">No history found</p>
              <p className="text-sm mt-1">Inferences run in the last 30 days will appear here.</p>
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-xs uppercase tracking-wider text-gray-500">
                  <th className="p-4 font-semibold">Date & Time</th>
                  <th className="p-4 font-semibold">Operator</th>
                  <th className="p-4 font-semibold">Part category</th>
                  <th className="p-4 font-semibold">Image name</th>
                  <th className="p-4 font-semibold">AI decision</th>
                  <th className="p-4 font-semibold text-right">Confidence score</th>
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
                           <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                           <span className="truncate max-w-37.5" title={record.imageName}>{record.imageName}</span>
                        </span>
                      ) : (
                        <span className="text-gray-300 italic">Unknown</span>
                      )}
                    </td>
                    <td className="p-4">
                      {record.isAnomaly ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">
                          <span className="w-1.5 h-1.5 rounded-full bg-red-600"></span> Anomaly
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-200">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-600"></span> Normal
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
      </div>
    </div>
  );
}