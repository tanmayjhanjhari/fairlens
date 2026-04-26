import { motion } from "framer-motion";
import { Check, Scan } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import useAnalysisStore from "../../store/analysisStore";

const STEPS = [
  { label: "Upload",     path: "/",           step: 0 },
  { label: "Configure",  path: "/analyze",    step: 1 },
  { label: "Analyze",    path: "/results",    step: 2 },
  { label: "Remediate",  path: "/remediate",  step: 3 },
  { label: "Report",     path: "/report-view",step: 4 },
];

const StepDot = ({ label, stepIndex, currentStep }) => {
  const isCompleted = currentStep > stepIndex;
  const isActive    = currentStep === stepIndex;
  const isFuture    = currentStep < stepIndex;

  return (
    <div className="flex items-center gap-2">
      <motion.div
        animate={{
          backgroundColor: isCompleted
            ? "#22C55E"
            : isActive
            ? "#14B8A6"
            : "#334155",
          scale: isActive ? 1.15 : 1,
        }}
        transition={{ duration: 0.25 }}
        className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
      >
        {isCompleted ? (
          <Check size={12} strokeWidth={3} className="text-white" />
        ) : (
          <span
            className={`text-[10px] font-bold ${
              isActive ? "text-primary" : "text-textSecondary"
            }`}
          >
            {stepIndex + 1}
          </span>
        )}
      </motion.div>

      <span
        className={`text-sm font-medium hidden sm:inline transition-colors duration-200 ${
          isActive
            ? "text-accent"
            : isCompleted
            ? "text-success"
            : "text-textSecondary"
        }`}
      >
        {label}
      </span>
    </div>
  );
};

const StepConnector = ({ completed }) => (
  <div className="flex-1 mx-2 h-px max-w-[48px]">
    <motion.div
      animate={{ backgroundColor: completed ? "#22C55E" : "#334155" }}
      transition={{ duration: 0.3 }}
      className="w-full h-full"
    />
  </div>
);

export default function AppShell() {
  const currentStep = useAnalysisStore((s) => s.step);
  const location    = useLocation();

  return (
    <header className="sticky top-0 z-50 bg-primary/95 backdrop-blur-md border-b border-white/[0.06]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between h-16">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center group-hover:bg-accent/30 transition-colors">
            <Scan size={18} className="text-accent" />
          </div>
          <span className="text-lg font-bold text-textPrimary">
            Fair<span className="text-accent">Lens</span>
          </span>
        </Link>

        {/* Step indicator */}
        <nav className="flex items-center" aria-label="Progress">
          {STEPS.map((s, idx) => (
            <div key={s.step} className="flex items-center">
              <StepDot
                label={s.label}
                stepIndex={s.step}
                currentStep={currentStep}
              />
              {idx < STEPS.length - 1 && (
                <StepConnector completed={currentStep > s.step} />
              )}
            </div>
          ))}
        </nav>
      </div>
    </header>
  );
}
