"use client";

import { useState, useEffect } from "react";

interface User {
  id: number;
  username: string;
  role: string;
}

export default function UsersManagementPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [updatingId, setUpdatingId] = useState<number | null>(null);

  // --- 1. FETCH ALL USERS ---
  const fetchUsers = async () => {
    setIsLoading(true);
    setErrorMessage("");
    try {
      const token = localStorage.getItem("token");
      const response = await fetch("https://localhost:7136/api/v1/Users", {
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setUsers(data);
      } else {
        setErrorMessage("Failed to load users. Are you sure you are logged in as an Admin?");
      }
    } catch (error) {
      setErrorMessage("Could not connect to the server.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  // --- 2. UPDATE USER ROLE ---
  const handleRoleChange = async (userId: number, newRole: string) => {
    setErrorMessage("");
    setSuccessMessage("");
    setUpdatingId(userId);

    try {
      const token = localStorage.getItem("token");
      const response = await fetch(`https://localhost:7136/api/v1/Users/${userId}/role`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ role: newRole })
      });

      if (response.ok) {
        setSuccessMessage(`Successfully updated permissions to ${newRole}!`);
        fetchUsers();
      } else {
        const errorText = await response.text();
        setErrorMessage(`Failed to update role: ${errorText}`);
      }
    } catch (error) {
      setErrorMessage("Network error while trying to update user role.");
    } finally {
      setUpdatingId(null);
    }
  };

  return (
    <div className="max-w-300 mx-auto space-y-8">
      
      {/* Page header */}
      <div className="mb-8 flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">User management</h2>
          <p className="text-gray-600 mt-1">Orchestrate system access and user roles.</p>
        </div>
        <button 
          onClick={fetchUsers}
          className="px-4 py-2 bg-white border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 flex items-center gap-2"
        >
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
          Refresh list
        </button>
      </div>

      {/* Alert messages */}
      {errorMessage && (
        <div className="p-4 bg-red-50 text-red-700 rounded-md border border-red-200 text-sm font-medium">
          {errorMessage}
        </div>
      )}
      {successMessage && (
        <div className="p-4 bg-green-50 text-green-700 rounded-md border border-green-200 text-sm font-medium">
          {successMessage}
        </div>
      )}

      {/* Users table */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-gray-400">
              <svg className="animate-spin h-8 w-8 mb-4 text-blue-500" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
              <p>Loading personnel data...</p>
            </div>
          ) : users.length === 0 ? (
            <div className="py-20 text-center text-gray-500">
              No users found in the database.
            </div>
          ) : (
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200 text-xs uppercase tracking-wider text-gray-500">
                  <th className="p-4 font-semibold w-24">User ID</th>
                  <th className="p-4 font-semibold">Username</th>
                  <th className="p-4 font-semibold">Current status</th>
                  <th className="p-4 font-semibold text-right">Manage role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 text-sm">
                {users.map((user) => (
                  <tr key={user.id} className="hover:bg-gray-50 transition-colors">
                    <td className="p-4 text-gray-500 font-mono">
                      #{user.id}
                    </td>
                    <td className="p-4">
                      <span className="font-medium text-gray-900">@{user.username}</span>
                    </td>
                    <td className="p-4">
                      {user.role === "Admin" ? (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-purple-100 text-purple-800 border border-purple-200">
                          Administrator
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-gray-100 text-gray-800 border border-gray-200">
                          Standard user
                        </span>
                      )}
                    </td>
                    <td className="p-4 text-right">
                      <div className="flex justify-end items-center gap-3">
                        {updatingId === user.id && (
                          <svg className="animate-spin h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        )}
                        <select
                          value={user.role}
                          onChange={(e) => handleRoleChange(user.id, e.target.value)}
                          disabled={updatingId === user.id}
                          className="rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-900 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 bg-white shadow-sm disabled:bg-gray-100 cursor-pointer"
                        >
                          <option value="Admin">Admin</option>
                          <option value="User">User</option>
                        </select>
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