import { motion } from "framer-motion";

export default function PageWrapper({ children, className = "" }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className={`max-w-7xl mx-auto px-4 sm:px-6 py-8 flex-1 ${className}`}
    >
      {children}
    </motion.div>
  );
}
