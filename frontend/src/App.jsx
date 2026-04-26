import { AnimatePresence, motion } from "framer-motion";
import { Toaster } from "react-hot-toast";
import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import AppShell from "./components/Layout/AppShell";
import AnalyzePage from "./pages/AnalyzePage";
import HomePage from "./pages/HomePage";
import RemediatePage from "./pages/RemediatePage";
import ReportPage from "./pages/ReportPage";
import ResultsPage from "./pages/ResultsPage";

// Page transition wrapper — applied per-route inside AnimatePresence
const PageTransition = ({ children }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -20 }}
    transition={{ duration: 0.3, ease: "easeInOut" }}
    style={{ flex: 1, display: "flex", flexDirection: "column" }}
  >
    {children}
  </motion.div>
);

// AnimatedRoutes reads location so AnimatePresence can detect route changes
const AnimatedRoutes = () => {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={
            <PageTransition>
              <HomePage />
            </PageTransition>
          }
        />
        <Route
          path="/analyze"
          element={
            <PageTransition>
              <AnalyzePage />
            </PageTransition>
          }
        />
        <Route
          path="/results"
          element={
            <PageTransition>
              <ResultsPage />
            </PageTransition>
          }
        />
        <Route
          path="/remediate"
          element={
            <PageTransition>
              <RemediatePage />
            </PageTransition>
          }
        />
        <Route
          path="/report-view"
          element={
            <PageTransition>
              <ReportPage />
            </PageTransition>
          }
        />
      </Routes>
    </AnimatePresence>
  );
};

export default function App() {
  return (
    <BrowserRouter>
      {/* Toast notifications — dark themed */}
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#1E293B",
            color: "#F1F5F9",
            border: "1px solid rgba(148,163,184,0.15)",
            fontFamily: "Space Grotesk, sans-serif",
            fontSize: "14px",
          },
          success: {
            iconTheme: { primary: "#22C55E", secondary: "#F1F5F9" },
          },
          error: {
            iconTheme: { primary: "#EF4444", secondary: "#F1F5F9" },
          },
        }}
      />

      {/* App shell (navbar + step indicator) */}
      <AppShell />

      {/* Animated page routes */}
      <AnimatedRoutes />
    </BrowserRouter>
  );
}
