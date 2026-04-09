"use client";

import { useState, useEffect } from "react";
import { jwtDecode } from "jwt-decode";

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

  const [currentUserId, setCurrentUserId] = useState<string | null>(null);
  const [updatingId, setUpdatingId] = useState<number | null>(null);

  const [deletingId, setDeletingId] = useState<number | null>(null);

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
    const token = localStorage.getItem("token");
    if (token) {
      try {
        const decoded: any = jwtDecode(token);
        const userId = decoded.id;
        
        setCurrentUserId(userId ? userId.toString() : null);
      } catch (e) {
        console.error("Failed to decode token", e);
      }
    }

    fetchUsers();
  }, []);

  // --- 2. UPDATE USER ROLE ---
  const handleRoleChange = async (userId: number, newRole: string) => {
    setUpdatingId(userId);
    setErrorMessage("");
    setSuccessMessage("");

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
        setSuccessMessage(`Successfully updated user to ${newRole}.`);
        fetchUsers(); 
      } else {
        const errorText = await response.text();
        setErrorMessage(errorText || "Failed to update role.");
        fetchUsers();
      }
    } catch (error) {
      setErrorMessage("Could not connect to the server to update role.");
      fetchUsers();
    } finally {
      setUpdatingId(null);
    }
  };

  // --- 3. DELETE USER ---
  const handleDeleteUser = async (id: number) => {
    if (!window.confirm("Are you sure you want to delete this user? This cannot be undone.")) {
      return;
    }

    setDeletingId(id);
    setErrorMessage("");
    setSuccessMessage("");

    try {
      const token = localStorage.getItem("token");
      const response = await fetch(`https://localhost:7136/api/v1/Users/${id}`, {
        method: "DELETE",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
      });

      if (response.ok) {
        setSuccessMessage("User deleted successfully.");
        setUsers(users.filter(user => user.id !== id));
      } else {
        const errorData = await response.text(); 
        setErrorMessage(errorData || "Failed to delete user.");
      }
    } catch (error) {
      setErrorMessage("Network error while trying to delete user.");
    } finally {
      setDeletingId(null);
    }
  };

  // --- UI RENDER ---
  return (
    <div className="max-w-5xl mx-auto space-y-6">
      
      {/* Page header */}
      <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200 flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">User management</h2>
          <p className="text-sm text-gray-500 mt-1">Orchestrate system access and user roles.</p>
        </div>
        <button 
          onClick={fetchUsers}
          disabled={isLoading}
          className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium rounded-lg transition-colors flex items-center gap-2"
        >
          {isLoading ? "Refreshing..." : "Refresh list"}
        </button>
      </div>

      {/* Messages */}
      {errorMessage && (
        <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-md shadow-sm">
          <p className="font-medium">Error</p>
          <p className="text-sm">{errorMessage}</p>
        </div>
      )}

      {successMessage && (
        <div className="p-4 bg-green-50 border-l-4 border-green-500 text-green-700 rounded-md shadow-sm">
          <p className="font-medium">Success</p>
          <p className="text-sm">{successMessage}</p>
        </div>
      )}

      {/* Users table */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
        {isLoading && users.length === 0 ? (
          <div className="p-12 text-center text-gray-500 flex flex-col items-center">
             <svg className="animate-spin h-8 w-8 text-blue-600 mb-4" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
            Loading users...
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  <th className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">ID</th>
                  <th className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">Username</th>
                  <th className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">Current role</th>
                  <th className="p-4 text-xs font-semibold text-gray-500 uppercase tracking-wider text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {users.map((user) => (
                  <tr key={user.id} className="hover:bg-gray-50 transition-colors">
                    <td className="p-4 text-sm text-gray-500">#{user.id}</td>
                    <td className="p-4 text-sm font-medium text-gray-900 flex items-center gap-2">
                      {user.username}
                      {user.id.toString() === currentUserId && (
                        <span className="text-xs font-bold text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full">
                          (You)
                        </span>
                      )}
                    </td>
                    <td className="p-4 text-sm">
                      <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                        user.role === 'Admin' ? 'bg-purple-100 text-purple-700 border border-purple-200' : 'bg-gray-100 text-gray-700 border border-gray-200'
                      }`}>
                        {user.role}
                      </span>
                    </td>
                    <td className="p-4 text-right">
                      <div className="flex justify-end items-center gap-3">
                        {updatingId === user.id && (
                          <svg className="animate-spin h-4 w-4 text-blue-600" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        )}
                        
                        {/* Role dropdown */}
                        <select
                          value={user.role}
                          onChange={(e) => handleRoleChange(user.id, e.target.value)}
                          disabled={updatingId === user.id || user.id.toString() === currentUserId}
                          className={`rounded-md border border-gray-300 px-3 py-1.5 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 shadow-sm transition-colors ${
                            user.id.toString() === currentUserId 
                              ? 'bg-gray-200 text-gray-500 cursor-not-allowed' 
                              : 'bg-white text-gray-900 cursor-pointer'
                          }`}
                        >
                          <option value="Admin">Admin</option>
                          <option value="User">User</option>
                        </select>
                        
                        <button
                          onClick={() => handleDeleteUser(user.id)}
                          disabled={deletingId === user.id || user.id.toString() === currentUserId}
                          className={`text-sm font-medium px-2 py-1.5 rounded transition-colors ${
                            user.id.toString() === currentUserId 
                              ? 'text-gray-400 cursor-not-allowed' 
                              : 'text-red-600 hover:text-red-800 hover:bg-red-50'
                          }`}
                        >
                          {deletingId === user.id ? "Deleting..." : "Delete"}
                        </button>
                      </div>

                    </td>
                  </tr>
                ))}
                
                {users.length === 0 && !isLoading && (
                  <tr>
                    <td colSpan={4} className="p-8 text-center text-gray-500">
                      No users found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}