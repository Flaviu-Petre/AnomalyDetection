"use client";

import { useState } from "react";
import Link from "next/link";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [message, setMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage("");
    setMessage("");
    setResetToken("");
    setIsLoading(true);

    try {
      const response = await fetch("https://localhost:7136/api/v1/Auth/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage(data.message);
        if (data.resetToken) {
          setResetToken(data.resetToken);
        }
      } else {
        setErrorMessage(data || "An error occurred.");
      }
    } catch {
      setErrorMessage("Could not connect to the server. Is the API running?");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-100">
      <div className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg">
        <h1 className="mb-2 text-2xl font-bold text-gray-800">Forgot password</h1>
        <p className="mb-6 text-sm text-gray-500">
          Enter your email address and a reset token will be generated.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-gray-700">
              Email address
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm focus:border-blue-500 focus:outline-none"
              placeholder="your@email.com"
            />
          </div>

          {errorMessage && (
            <p className="text-sm text-red-600">{errorMessage}</p>
          )}

          {message && (
            <div className="rounded-lg bg-green-50 p-3 text-sm text-green-700">
              <p>{message}</p>
              {resetToken && (
                <div className="mt-2">
                  <p className="font-medium">Your reset token:</p>
                  <code className="mt-1 block break-all rounded bg-gray-100 p-2 text-xs text-gray-800">
                    {resetToken}
                  </code>
                  <p className="mt-1 text-xs text-gray-500">
                    Copy this token and use it on the reset password page. It expires in 1 hour.
                  </p>
                </div>
              )}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full rounded-lg bg-blue-600 py-2 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-50"
          >
            {isLoading ? "Sending..." : "Generate reset token"}
          </button>
        </form>

        <div className="mt-4 text-center text-sm">
          <Link href="/reset-password" className="font-semibold text-blue-600 hover:text-blue-500">
            I already have a token.
          </Link>
        </div>

        <div className="mt-2 text-center text-sm">
          <span className="text-gray-500">Remember your password? </span>
          <Link href="/login" className="font-semibold text-blue-600 hover:text-blue-500">
            Sign in here.
          </Link>
        </div>
      </div>
    </main>
  );
}