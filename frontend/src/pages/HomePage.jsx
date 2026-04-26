import { motion } from "framer-motion";
import { ArrowRight, BarChart3, Brain, FileSearch, Shield, Zap } from "lucide-react";
import { useNavigate } from "react-router-dom";
import PageWrapper from "../components/Layout/PageWrapper";
import useAnalysisStore from "../store/analysisStore";

const features = [
  { icon: FileSearch, title: "CSV + Model Upload",   desc: "Drag-and-drop your dataset and ML model. Instant column detection and type analysis." },
  { icon: BarChart3,  title: "Fairness Metrics",     desc: "SPD, DI, EOD, AOD computed per sensitive attribute with severity grading." },
  { icon: Brain,      title: "Gemini 2.0 Flash AI",  desc: "Scenario auto-detection and plain-English bias explanations for any audience." },
  { icon: Shield,     title: "Bias Audit Score",     desc: "0–100 composite score graded A/B/C/F with legal threshold flagging (DI < 0.8)." },
  { icon: Zap,        title: "Auto Mitigation",      desc: "Reweighing and threshold adjustment run simultaneously with trade-off analysis." },
  { icon: FileSearch, title: "PDF Audit Report",     desc: "Professional multi-page report ready for compliance officers and model cards." },
];

export default function HomePage() {
  const navigate  = useNavigate();
  const setStep   = useAnalysisStore((s) => s.setStep);

  const handleStart = () => {
    setStep(0);
    navigate("/analyze");
  };

  return (
    <PageWrapper>
      {/* Hero */}
      <div className="text-center py-16">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 bg-accent/10 border border-accent/30 text-accent text-sm font-medium px-4 py-1.5 rounded-full mb-6"
        >
          <span className="w-2 h-2 rounded-full bg-accent animate-pulse" />
          Powered by Google Gemini 2.0 Flash
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-5xl sm:text-6xl font-bold text-textPrimary mb-6 leading-tight"
        >
          Detect. Explain.{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent to-accent2">
            Mitigate.
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="text-xl text-textSecondary max-w-2xl mx-auto mb-10"
        >
          FairLens is an AI-powered bias auditing platform. Upload your dataset,
          get fairness metrics, plain-English explanations, and automated mitigation
          — in under 3 minutes.
        </motion.p>

        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          onClick={handleStart}
          className="btn-primary inline-flex items-center gap-2 text-base px-8 py-3"
        >
          Start Bias Audit
          <ArrowRight size={18} />
        </motion.button>
      </div>

      {/* Feature grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 mt-8">
        {features.map((f, i) => (
          <motion.div
            key={f.title}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 * i }}
            className="glass-card p-6 hover:border-accent/30 transition-colors duration-300"
          >
            <div className="w-10 h-10 rounded-lg bg-accent/15 flex items-center justify-center mb-4">
              <f.icon size={20} className="text-accent" />
            </div>
            <h3 className="text-base font-semibold text-textPrimary mb-2">{f.title}</h3>
            <p className="text-sm text-textSecondary leading-relaxed">{f.desc}</p>
          </motion.div>
        ))}
      </div>
    </PageWrapper>
  );
}
