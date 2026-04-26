import axios from "axios";
import toast from "react-hot-toast";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  timeout: 60000,
  headers: {
    "Content-Type": "application/json",
  },
});

// ── Response interceptor ──────────────────────────────────────────────────────
client.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail =
      error?.response?.data?.detail ||
      error?.response?.data?.message ||
      error?.message ||
      "Something went wrong. Please try again.";

    // Don't toast for 404 on certain polling calls — caller can handle those
    const status = error?.response?.status;
    if (status !== 404) {
      toast.error(detail, {
        duration: 5000,
        style: {
          background: "#1E293B",
          color: "#F1F5F9",
          border: "1px solid rgba(239,68,68,0.4)",
        },
        iconTheme: { primary: "#EF4444", secondary: "#F1F5F9" },
      });
    }

    return Promise.reject(error);
  }
);

export default client;
