"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [userRole, setUserRole] = useState<string | null>(null);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
      return;
    }

    const role = localStorage.getItem("role");
    setUserRole(role);
  }, [router]);

  const isActive = (path: string) => pathname === path;

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    router.push("/login");
  };

  return (
    <div className="flex h-screen bg-gray-100">

      {/* GLOBAL SIDEBAR */}
      <aside className="w-64 bg-blue-900 text-white flex flex-col">
        <div className="p-6">
          <h1 className="text-2xl font-bold tracking-wider">FactoryOS</h1>
          <p className="text-blue-300 text-sm mt-1">
            {userRole === "Admin" ? "Admin portal" : "Operator portal"}
          </p>
        </div>

        <nav className="flex-1 px-4 space-y-2 mt-4">
          <Link href="/dashboard" className={`block px-4 py-2 rounded-md transition-colors ${pathname === '/dashboard' ? 'bg-blue-800 font-medium' : 'text-blue-200 hover:bg-blue-800 hover:text-white'}`}>
            Dashboard
          </Link>
          <Link href="/dashboard/inference" className={`block px-4 py-2 rounded-md transition-colors ${pathname === '/dashboard/inference' ? 'bg-blue-800 font-medium' : 'text-blue-200 hover:bg-blue-800 hover:text-white'}`}>
            Run inference
          </Link>
          <Link href="/dashboard/history" className={`block px-4 py-2 rounded-md transition-colors ${pathname === '/dashboard/history' ? 'bg-blue-800 font-medium' : 'text-blue-200 hover:bg-blue-800 hover:text-white'}`}>
            Inference history
          </Link>

          {/* Only show model manager and user manager to admins */}
          {userRole === "Admin" && (
            <>
              <Link href="/dashboard/models" className={`block px-4 py-2 rounded-md transition-colors ${pathname === '/dashboard/models' ? 'bg-blue-800 font-medium' : 'text-blue-200 hover:bg-blue-800 hover:text-white'}`}>
                Model manager
              </Link>
              <Link href="/dashboard/users" className={`block px-4 py-2 rounded-md transition-colors ${pathname === '/dashboard/users' ? 'bg-blue-800 font-medium' : 'text-blue-200 hover:bg-blue-800 hover:text-white'}`}>
                Manage users
              </Link>
            </>
          )}
        </nav>

        <div className="p-4 border-t border-blue-800">
          <button
            onClick={handleLogout}
            className="w-full text-left px-4 py-2 text-red-300 hover:bg-blue-800 hover:text-red-100 rounded-md transition-colors"
          >
            Sign Out
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 flex flex-col overflow-hidden">

        {/* GLOBAL TOP HEADER */}
        <header className="bg-white shadow-sm z-10 p-4 flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-800">
            {pathname === '/dashboard' ? 'System overview' :
              pathname === '/dashboard/inference' ? 'Run inference' :
                pathname === '/dashboard/history' ? 'Inference history' :
                  pathname === '/dashboard/models' ? 'Model manager' :
                    pathname === '/dashboard/users' ? 'User management' : ''}
          </h2>
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-gray-600 bg-gray-100 px-3 py-1 rounded-full border border-gray-200">
              Role: <span className="text-blue-600 font-bold">{userRole}</span>
            </span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8">
          {children}
        </div>
      </main>

    </div>
  );
}